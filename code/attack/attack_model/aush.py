import torch
import torch.nn as nn
import math
from .base_attacker import BaseAttacker
import numpy as np
import pandas as pd
from functools import partial
from rectool import pick_optim, filler_filter_mat



class AUSH(BaseAttacker):
    def __init__(self, train_df,config ):
        super().__init__( train_df,config)
        self.selected_ids = config["attack_model_p"]['selected_ids']
        print("selected_ids",self.selected_ids,type(self.selected_ids))
        self.selected_index_ids = np.array(np.where(np.isin(self.item_unique, self.selected_ids))[0])
        print("selected_index_ids",self.selected_index_ids)
        self.selected_ids = self.selected_index_ids
        self.config = config
        self.device = config['device']
        self.ZR_ratio = config["attack_model_p"]['ZR_ratio']
        # self.dataset = dfDataset(train_df,config)


        self.train_data_array = self.dataset.to_matrix()   # csr matrix

        self.build_network()

    def build_network(self):
        self.netG = AushGenerator(self.config['task_type'], input_dim=self.item_num).to(self.device)
        self.G_optimizer = pick_optim(self.config["attack_model_p"]['optim_g'])(
            self.netG.parameters(), lr=self.config["attack_model_p"]['lr_g']
        )

        self.netD = AushDiscriminator(input_dim=self.item_num).to(self.device)
        self.D_optimizer = pick_optim(self.config["attack_model_p"]['optim_d'])(
            self.netD.parameters(), lr=self.config["attack_model_p"]['lr_d']
        ) 

    def sample_fillers(self, real_profiles, target_id_list):
        # 随机选择filler_num个未评分的item作为填充
        fillers = np.zeros_like(real_profiles)
        filler_pool = (
            set(range(self.item_num)) - set(self.selected_ids) - set(target_id_list)
        )
        filler_sampler = lambda x: np.random.choice(
            size=self.filler_num,
            replace=True,
            a=list(set(np.argwhere(x > 0).flatten()) & filler_pool),
            # a=list(set(filler_pool)), # 可能存在数据问题
        )

        sampled_cols = np.array(
            [filler_sampler(x) for x in real_profiles], dtype="int64"
        )

        sampled_rows = np.repeat(np.arange(real_profiles.shape[0]), self.filler_num)
        fillers[sampled_rows, sampled_cols.flatten()] = 1
        return fillers
    
    def train_step(self):
        target_id_list = self.target_id_list
        bce_loss = nn.BCELoss()
        mse_loss = nn.MSELoss()

        g_loss_rec_l = []
        g_loss_shilling_l = []
        g_loss_gan_l = []
        d_losses = []

        index_filter = partial(
            filler_filter_mat,
            selected_ids=self.selected_ids,
            filler_num=self.filler_num,
            target_id_list=target_id_list,
        )
        
        for idx, dp in enumerate(
            self.dataset.generate_batch(user_filter=index_filter)
        ):
            batch_set_idx = dp['users']
            real_profiles = dp['users_mat']

            valid_labels = (
                torch.ones_like(batch_set_idx)
                .type(torch.float)
                .to(self.device)
                .reshape(len(batch_set_idx), 1)
            )
            fake_labels = (
                torch.zeros_like(batch_set_idx)
                .type(torch.float)
                .to(self.device)
                .reshape(len(batch_set_idx), 1)
            )
            fillers_mask = self.sample_fillers(
                real_profiles.cpu().numpy(), target_id_list
            )

            # selected
            selects_mask = np.zeros_like(fillers_mask)
            selects_mask[:, self.selected_ids] = 1.0
            # target
            target_patch = np.zeros_like(fillers_mask)
            target_patch[:, self.selected_ids] = 5.0
            # ZR_mask
            ZR_mask = (real_profiles.cpu().numpy() == 0) * selects_mask
            pools = np.argwhere(ZR_mask)
            np.random.shuffle(pools)
            pools = pools[: math.floor(len(pools) * (1 - self.ZR_ratio))]
            ZR_mask[pools[:, 0], pools[:, 1]] = 0

            fillers_mask = torch.tensor(fillers_mask).type(torch.float).to(self.device)
            selects_mask = torch.tensor(selects_mask).type(torch.float).to(self.device)
            target_patch = torch.tensor(target_patch).type(torch.float).to(self.device)
            ZR_mask = torch.tensor(ZR_mask).type(torch.float).to(self.device)

            input_template = torch.mul(real_profiles, fillers_mask)
            # ----------generate----------
            self.netG.eval()
            gen_output = self.netG(input_template)
            gen_output = gen_output.detach()
            # ---------mask--------
            selected_patch = torch.mul(gen_output, selects_mask)
            middle = torch.add(input_template, selected_patch)
            fake_profiles = torch.add(middle, target_patch)
            # --------Discriminator------
            # forward
            self.D_optimizer.zero_grad()
            self.netD.train()
            d_valid_labels = self.netD(real_profiles * (fillers_mask + selects_mask))
            d_fake_labels = self.netD(fake_profiles * (fillers_mask + selects_mask))
            # loss
            D_real_loss = bce_loss(d_valid_labels, valid_labels)
            D_fake_loss = bce_loss(d_fake_labels, fake_labels)
            d_loss = 0.5 * (D_real_loss + D_fake_loss)
            d_loss.backward()
            self.D_optimizer.step()
            self.netD.eval()
            d_losses.append(d_loss.item())

            # ---------train G-------
            self.netG.train()
            d_fake_labels = self.netD(fake_profiles * (fillers_mask + selects_mask))
            g_loss_gan = bce_loss(d_fake_labels, valid_labels)
            g_loss_shilling = mse_loss(fake_profiles * selects_mask, selects_mask * 5.0)
            g_loss_rec = mse_loss(
                fake_profiles * selects_mask * ZR_mask,
                selects_mask * input_template * ZR_mask,
            )
            g_loss = g_loss_gan + g_loss_rec + g_loss_shilling

            g_loss_rec_l.append(g_loss_rec.item())
            g_loss_shilling_l.append(g_loss_shilling.item())
            g_loss_gan_l.append(g_loss_gan.item())
            self.G_optimizer.zero_grad()
            g_loss.backward(retain_graph=True)
            self.G_optimizer.step()
        return (
            np.mean(d_losses),
            np.mean(g_loss_rec_l),
            np.mean(g_loss_shilling_l),
            np.mean(g_loss_gan_l),
        )
    
    def train(self):
        for epoch in range(self.config["attack_model_p"]['attack_epoch']):
            d_loss, g_loss_rec, g_loss_shilling, g_loss_gan = self.train_step()
            print(
                f"Epoch: {epoch}, D_loss: {d_loss}, G_loss_rec: {g_loss_rec}, G_loss_shilling: {g_loss_shilling}, G_loss_gan: {g_loss_gan}"
            )
        print("Training Done!")
        # return self.netG
        



    def get_explicit_fake_df(self):
        self.train()
        target_id_list = self.target_id_list

        mask_array = (self.train_data_array > 0).astype('float') # 评分矩阵转化为01矩阵；0表示未评分，1表示已评分
        mask_array[:, np.concatenate([self.selected_ids , target_id_list])] = 0
        available_idx = np.where(np.sum(mask_array, 1) >= self.filler_num)[0] # 在mask_array数据中选择所有剩余交互item数量大于filler_num的用户id
        available_idx = np.random.permutation(available_idx) # 打乱用户id
        idx = available_idx[np.random.randint(0, len(available_idx), self.attacker_num)] # 随机选择attack_num个用户id
        idx = list(idx)

        real_profiles = self.train_data_array[idx, :]
        # sample fillers
        fillers_mask = self.sample_fillers(real_profiles, target_id_list)

        # selected
        selects_mask = np.zeros_like(fillers_mask)
        selects_mask[:, self.selected_ids] = 1.0
        # target
        target_patch = np.zeros_like(fillers_mask)
        target_patch[:, target_id_list] = 5.0

        # Generate
        real_profiles = torch.tensor(real_profiles).type(torch.float).to(self.device)
        fillers_mask = torch.tensor(fillers_mask).type(torch.float).to(self.device)
        selects_mask = torch.tensor(selects_mask).type(torch.float).to(self.device)
        target_patch = torch.tensor(target_patch).type(torch.float).to(self.device)
        input_template = torch.mul(real_profiles, fillers_mask)
        self.netG.eval()
        gen_output = self.netG(input_template)
        selected_patch = torch.mul(gen_output, selects_mask)
        middle = torch.add(input_template, selected_patch) # netG中生成的数据中仅利用selected部分
        fake_profiles = torch.add(middle, target_patch) # target item部分为手动设置的
        fake_profiles = fake_profiles.detach().cpu().numpy()
        # fake_profiles = self.generator.predict([real_profiles, fillers_mask, selects_mask, target_patch])
        # selected patches
        selected_patches = fake_profiles[:, self.selected_ids]
        selected_patches = np.round(selected_patches)
        selected_patches[selected_patches > 5] = 5
        selected_patches[selected_patches < 1] = 1
        fake_profiles[:, self.selected_ids] = selected_patches


        fake_user_id_start = self.train_df[self._USER].max() + 1
        fake_user_id_list = []
        # TODO:fake user 数量存在问题
        for i in range(self.attacker_num):
            fake_user_id_list.extend([fake_user_id_start+i])
        fake_df = self._fake_profile2df(fake_profiles,fake_user_id_list)
        return fake_df

    def get_implicit_fake_df(self):
        self.train()
        target_id_list = self.target_id_list

        mask_array = (self.train_data_array > 0).astype('float') # 评分矩阵转化为01矩阵；0表示未评分，1表示已评分
        mask_array[:, np.concatenate([self.selected_ids , target_id_list])] = 0
        available_idx = np.where(np.sum(mask_array, 1) >= self.filler_num)[0] # 在mask_array数据中选择所有剩余交互item数量大于filler_num的用户id
        available_idx = np.random.permutation(available_idx) # 打乱用户id
        idx = available_idx[np.random.randint(0, len(available_idx), self.attacker_num)] # 随机选择attack_num个用户id
        idx = list(idx)

        real_profiles = self.train_data_array[idx, :]
        # sample fillers
        fillers_mask = self.sample_fillers(real_profiles, target_id_list)

        # selected
        selects_mask = np.zeros_like(fillers_mask)
        selects_mask[:, self.selected_ids] = 1.0
        # target
        target_patch = np.zeros_like(fillers_mask)
        target_patch[:, target_id_list] = 1.0

        # Generate
        real_profiles = torch.tensor(real_profiles).type(torch.float).to(self.device)
        fillers_mask = torch.tensor(fillers_mask).type(torch.float).to(self.device)
        selects_mask = torch.tensor(selects_mask).type(torch.float).to(self.device)
        target_patch = torch.tensor(target_patch).type(torch.float).to(self.device)
        input_template = torch.mul(real_profiles, fillers_mask)
        self.netG.eval()
        gen_output = self.netG(input_template)
        selected_patch = torch.mul(gen_output, selects_mask)
        middle = torch.add(input_template, selected_patch) # netG中生成的数据中仅利用selected部分
        fake_profiles = torch.add(middle, target_patch) # target item部分为手动设置的
        fake_profiles = fake_profiles.detach().cpu().numpy()
        # fake_profiles = self.generator.predict([real_profiles, fillers_mask, selects_mask, target_patch])
        # selected patches
        selected_patches = fake_profiles[:, self.selected_ids]
        selected_patches = np.round(selected_patches)
        selected_patches[selected_patches > 0.1] = 1
        selected_patches[selected_patches <= 0.1] = 0
        fake_profiles[:, self.selected_ids] = selected_patches


        fake_user_id_start = self.train_df[self._USER].max() + 1
        fake_user_id_list = []
        # TODO:fake user 数量存在问题
        for i in range(self.attacker_num):
            fake_user_id_list.extend([fake_user_id_start+i])
        fake_df = self._fake_profile2df(fake_profiles,fake_user_id_list)
        return fake_df


class AushGenerator(nn.Module):
    def __init__(self,task_type, input_dim):
        super(AushGenerator, self).__init__()
        self.task_type = task_type
        self.main = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.Sigmoid(),
            nn.Linear(128, input_dim),
            nn.Sigmoid(),
        )

    def forward(self, input):
        if self.task_type == 'rating':
            return self.main(input) * 5
        elif self.task_type == 'ranking':
            return self.main(input)

class AushDiscriminator(nn.Module):
    def __init__(self, input_dim):
        super(AushDiscriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(input_dim, 150),
            nn.Sigmoid(),
            nn.Linear(150, 150),
            nn.Sigmoid(),
            nn.Linear(150, 150),
            nn.Sigmoid(),
            nn.Linear(150, 1),
            nn.Sigmoid(),
        )

    def forward(self, input):
        return self.main(input)
