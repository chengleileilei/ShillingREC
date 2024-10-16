import numpy as np
import torch
import torch.nn as nn
import rectool
from data import MemLoader, Loader
from rectool import timer
# import model
from model.rec.sgdl import LightGCN, LTW

import multiprocessing
from sklearn.mixture import GaussianMixture as GMM
from copy import deepcopy
from torch.distributions.categorical import Categorical

CORES = multiprocessing.cpu_count() // 2

class Scheduler(nn.Module):
    def __init__(self, N):
        super(Scheduler, self).__init__()
        self.grad_lstm = nn.LSTM(N, 10, 1, bidirectional=True)
        self.loss_lstm = nn.LSTM(1, 10, 1, bidirectional=True)
        self.cosine = torch.nn.CosineSimilarity(dim=-1, eps=1e-8)
        input_dim = 40
        self.fc1 = nn.Linear(input_dim, 20)
        self.fc2 = nn.Linear(20, 1)

    def forward(self, loss, input):
        grad_output, (hn, cn) = self.grad_lstm(input.reshape(1, len(input), -1))
        grad_output = grad_output.sum(0)

        loss_output, (hn, cn) = self.loss_lstm(loss.reshape(1, len(loss), 1))
        loss_output = loss_output.sum(0)

        x = torch.cat((grad_output, loss_output), dim=1)

        z = torch.tanh(self.fc1(x))
        z = self.fc2(z)
        return z

    def sample_task(self, prob, size, replace=True):
        self.m = Categorical(prob)
        p = prob.detach().cpu().numpy()
        if len(np.where(p > 0)[0]) < size:
            actions = torch.tensor(np.where(p > 0)[0])
        else:
            actions = np.random.choice(np.arange(len(prob)), p=p / np.sum(p), size=size,
                                       replace=replace)
            actions = [torch.tensor(x).cuda() for x in actions]
        return torch.LongTensor(actions)

    def sample_gumbel(self, shape, eps=1e-20):
        """Sample from Gumbel(0, 1)"""
        U = torch.rand(shape)
        return -torch.log(-torch.log(U + eps) + eps).cuda()

    def gumbel_softmax_sample(self, logits, temperature):
        """ Draw a sample from the Gumbel-Softmax distribution"""
        y = logits + self.sample_gumbel(logits.shape)
        return torch.softmax(y / temperature, dim=-1)

    def gumbel_softmax(self, logits, temperature, hard=False):
        """Sample from the Gumbel-Softmax distribution and optionally discretize.
        Args:
          logits: [batch_size, n_class] unnormalized log-probs
          temperature: non-negative scalar
          hard: if True, take argmax, but differentiate w.r.t. soft sample y
        Returns:
          [batch_size, n_class] sample from the Gumbel-Softmax distribution.
          If hard=True, then the returned sample will be one-hot, otherwise it will
          be a probabilitiy distribution that sums to 1 across classes
        """
        y = self.gumbel_softmax_sample(logits, temperature)
        if hard:
            y_hard = torch.eq(y, torch.max(y, 1, keepdim=True).values).long().cuda()
            y = (y_hard - y).detach() + y
            #y = torch.nonzero(y)[:, 1]
        return y

def memorization_train(config, dataset, recommend_model, opt):
    Recmodel = recommend_model
    Recmodel.train()

    # sampling
    S = rectool.UniformSample(dataset)
    users = torch.Tensor(S[:, 0]).long()
    posItems = torch.Tensor(S[:, 1]).long()
    negItems = torch.Tensor(S[:, 2]).long()

    users = users.to(config['device'])
    posItems = posItems.to(config['device'])
    negItems = negItems.to(config['device'])
    users, posItems, negItems = rectool.shuffle(users, posItems, negItems)
    total_batch = len(users) // config["rec_model_p"]["batch_size"] + 1
    aver_loss = 0.
    for (batch_i,
         (batch_users,
          batch_pos,
          batch_neg)) in enumerate(rectool.minibatch(users,
                                                   posItems,
                                                   negItems,
                                                   batch_size=config["rec_model_p"]["batch_size"])):

        loss, reg_loss = Recmodel.loss(batch_users, batch_pos, batch_neg)
        opt.zero_grad()
        loss.backward()
        opt.step()
        aver_loss += loss.cpu().item()

    aver_loss = aver_loss / total_batch
    timer.zero()
    return f"{aver_loss:.5f}"

def estimate_noise(config, dataset, recommend_model):
    '''
    estimate noise ratio based on GMM
    '''
    Recmodel: LightGCN = recommend_model
    Recmodel.eval()

    dataset: MemLoader

    # sampling
    S = rectool.UniformSample(dataset)
    users_origin = torch.Tensor(S[:, 0]).long()
    posItems_origin = torch.Tensor(S[:, 1]).long()
    negItems_origin = torch.Tensor(S[:, 2]).long()

    users_origin = users_origin.to(config['device'])
    posItems_origin = posItems_origin.to(config['device'])
    negItems_origin = negItems_origin.to(config['device'])
    with torch.no_grad():
        losses = []
        for (batch_i,
             (batch_users,
              batch_pos,
              batch_neg)) in enumerate(rectool.minibatch(users_origin,
                                                       posItems_origin,
                                                       negItems_origin,
                                                       batch_size=config["rec_model_p"]["batch_size"])):
            loss, _ = Recmodel.loss(batch_users, batch_pos, batch_neg, reduce=False)
            # concat all losses
            if len(losses) == 0:
                losses = loss
            else:
                losses = torch.cat((losses, loss), dim=0)
        # split losses of each user
        losses_u = []
        st, ed = 0, 0
        for count in dataset.user_pos_counts:
            ed = st + count
            losses_u.append(losses[st:ed])
            st = ed
        # normalize losses of each user
        for i in range(len(losses_u)):
            if len(losses_u[i]) > 1:
                losses_u[i] = (losses_u[i] - losses_u[i].min()) / (losses_u[i].max() - losses_u[i].min())
        losses = torch.cat(losses_u, dim=0)
        losses = losses.reshape(-1, 1).cpu().detach().numpy()
        gmm = GMM(n_components=2, max_iter=100, tol=1e-2, reg_covar=5e-4)
        gmm.fit(losses)
        prob = gmm.predict_proba(losses)
        prob = prob[:, gmm.means_.argmax()]
        return 1 - np.mean(prob)


def self_guided_train_schedule_reinforce(config, train_dataset, clean_dataset, recmodel, ltw:LTW):
    train_loss, meta_loss = 0, 0
    scheduler = Scheduler(len(recmodel.state_dict())).cuda()
    recmodel.train()
    train_opt = torch.optim.Adam(recmodel.params(), lr=config["rec_model_p"]["lr"])
    meta_opt = torch.optim.Adam(ltw.params(), lr=config["rec_model_p"]["meta_lr"])
    schedule_opt = torch.optim.Adam(scheduler.parameters(), lr=config["rec_model_p"]["schedule_lr"])

    # sampling
    with timer(name='Train Sample'):
        train_data = rectool.UniformSample(train_dataset)
    with timer(name='Clean Sample'):
        clean_data = rectool.UniformSample(clean_dataset)

    users = torch.Tensor(train_data[:, 0]).long().to(config['device'])
    posItems = torch.Tensor(train_data[:, 1]).long().to(config['device'])
    negItems = torch.Tensor(train_data[:, 2]).long().to(config['device'])

    users_clean = torch.Tensor(clean_data[:, 0]).long().to(config['device'])
    posItems_clean = torch.Tensor(clean_data[:, 1]).long().to(config['device'])
    negItems_clean = torch.Tensor(clean_data[:, 2]).long().to(config['device'])

    users, posItems, negItems = rectool.shuffle(users, posItems, negItems)
    users_clean, posItems_clean, negItems_clean = rectool.shuffle(users_clean, posItems_clean, negItems_clean)

    total_batch = len(users) // config["rec_model_p"]["batch_size"] + 1

    clean_data_iter = iter(
        rectool.minibatch(users_clean, posItems_clean, negItems_clean, batch_size=config["rec_model_p"]["batch_size"]))
    for batch_i, (batch_users, batch_pos, batch_neg) in enumerate(rectool.minibatch(users,
                                                                                  posItems,
                                                                                  negItems,
                                                                                  batch_size=config["rec_model_p"]["batch_size"])):

        try:
            batch_users_clean, batch_pos_clean, batch_neg_clean = next(clean_data_iter)
        except StopIteration:
            clean_data_iter = iter(rectool.minibatch(users_clean,
                                                   posItems_clean,
                                                   negItems_clean,
                                                   batch_size=config["rec_model_p"]["batch_size"]))
            batch_users_clean, batch_pos_clean, batch_neg_clean = next(clean_data_iter)

        meta_model = deepcopy(recmodel)

        # ============= get input of the scheduler ============= #
        L_theta, _ = meta_model.loss(batch_users_clean, batch_pos_clean, batch_neg_clean, reduce=False)
        L_theta = torch.reshape(L_theta, (len(L_theta), 1))

        grads_theta_list = []
        for k in range(len(batch_users_clean)):
            grads_theta_list.append(torch.autograd.grad(L_theta[k], (meta_model.params()), create_graph=True,
                                                        retain_graph=True))

        v_L_theta = ltw(L_theta.data)

        # assumed update
        L_theta_meta = torch.sum(L_theta * v_L_theta) / len(batch_users_clean)
        meta_model.zero_grad()
        grads = torch.autograd.grad(L_theta_meta, (meta_model.params()), create_graph=True, retain_graph=True)

        meta_model.update_params(lr_inner=config["rec_model_p"]["lr"], source_params=grads)
        del grads

        L_theta_hat, _ = meta_model.loss(batch_users_clean, batch_pos_clean, batch_neg_clean, reduce=False)

        # for each sample, calculate gradients of 2 losses
        input_embedding_cos = []
        for k in range(len(batch_users_clean)):
            task_grad_cos = []
            grads_theta = grads_theta_list[k]
            grads_theta_hat = torch.autograd.grad(L_theta_hat[k], (meta_model.params()), create_graph=True,
                                                  retain_graph=True)

            # calculate cosine similarity for each parameter
            for j in range(len(grads_theta)):
                task_grad_cos.append(scheduler.cosine(grads_theta[j].flatten().unsqueeze(0),
                                                      grads_theta_hat[j].flatten().unsqueeze(0))[0])
            del grads_theta
            del grads_theta_hat
            # stack similarity of each parameter
            task_grad_cos = torch.stack(task_grad_cos)
            # stack similarity of each sample
            input_embedding_cos.append(task_grad_cos.detach())

        # sample clean data
        weight = scheduler(L_theta, torch.stack(input_embedding_cos).cuda())
        task_prob = torch.softmax(weight.reshape(-1), dim=-1)
        sample_idx = scheduler.sample_task(task_prob, len(batch_users_clean))
        batch_users_clean = batch_users_clean[sample_idx]
        batch_pos_clean = batch_pos_clean[sample_idx]
        batch_neg_clean = batch_neg_clean[sample_idx]

        # ============= training ============= #
        meta_model = deepcopy(recmodel)

        # assumed update of theta (theta -> theta')
        cost, reg_loss = meta_model.loss(batch_users, batch_pos, batch_neg, reduce=False)
        cost_v = torch.reshape(cost, (len(cost), 1))
        v_lambda = ltw(cost_v.data)

        l_f_meta = torch.sum(cost_v * v_lambda) / len(batch_users)
        meta_model.zero_grad()
        grads = torch.autograd.grad(l_f_meta, (meta_model.params()), create_graph=True)

        # load theta' and update params of ltw
        meta_model.update_params(lr_inner=config["rec_model_p"]["lr"], source_params=grads)
        del grads

        l_g_meta, _ = meta_model.loss(batch_users_clean, batch_pos_clean, batch_neg_clean)

        # REINFORCE
        loss_schedule = 0
        for idx in sample_idx:
            loss_schedule += scheduler.m.log_prob(idx.cuda())
        reward = l_g_meta
        loss_schedule *= reward

        meta_opt.zero_grad()
        l_g_meta.backward(retain_graph=True)
        meta_opt.step()

        schedule_opt.zero_grad()
        loss_schedule.backward()
        schedule_opt.step()

        # reload and actually update theta
        cost_w, _ = recmodel.loss(batch_users, batch_pos, batch_neg, reduce=False)
        cost_v = torch.reshape(cost_w, (len(cost_w), 1))
        with torch.no_grad():
            w_new = ltw(cost_v)
        loss = torch.sum(cost_v * w_new) / len(batch_users)

        train_opt.zero_grad()
        loss.backward()
        train_opt.step()

        recmodel.store_params()

        train_loss += loss.cpu().item()
        meta_loss += l_g_meta.cpu().item()

    train_loss /= total_batch
    meta_loss /= total_batch
    timer.zero()
    return [f'{train_loss:.5f}', f'{meta_loss:.5f}']


def self_guided_train_schedule_gumbel(config, train_dataset, clean_dataset, recmodel, ltw:LTW):
    train_loss, meta_loss = 0, 0
    scheduler = Scheduler(len(recmodel.state_dict())).cuda()
    recmodel.train()

    train_opt = torch.optim.Adam(recmodel.params(), lr=config["rec_model_p"]["lr"])
    meta_opt = torch.optim.Adam(ltw.params(), lr=config["rec_model_p"]["meta_lr"])
    schedule_opt = torch.optim.Adam(scheduler.parameters(), lr=config["rec_model_p"]["schedule_lr"])

    # sampling
    train_data = rectool.UniformSample(train_dataset)
    clean_data = rectool.UniformSample(clean_dataset)

    users = torch.Tensor(train_data[:, 0]).long().to(config['device'])
    posItems = torch.Tensor(train_data[:, 1]).long().to(config['device'])
    negItems = torch.Tensor(train_data[:, 2]).long().to(config['device'])

    users_clean = torch.Tensor(clean_data[:, 0]).long().to(config['device'])
    posItems_clean = torch.Tensor(clean_data[:, 1]).long().to(config['device'])
    negItems_clean = torch.Tensor(clean_data[:, 2]).long().to(config['device'])

    users, posItems, negItems = rectool.shuffle(users, posItems, negItems)
    users_clean, posItems_clean, negItems_clean = rectool.shuffle(users_clean, posItems_clean, negItems_clean)

    total_batch = len(users) // config["rec_model_p"]["batch_size"] + 1

    clean_data_iter = iter(
        rectool.minibatch(users_clean, posItems_clean, negItems_clean, batch_size=config["rec_model_p"]["batch_size"]))
    for batch_i, (batch_users, batch_pos, batch_neg) in enumerate(rectool.minibatch(users,
                                                                                  posItems,
                                                                                  negItems,
                                                                                  batch_size=config["rec_model_p"]["batch_size"])):

        try:
            batch_users_clean, batch_pos_clean, batch_neg_clean = next(clean_data_iter)
        except StopIteration:
            clean_data_iter = iter(rectool.minibatch(users_clean,
                                                   posItems_clean,
                                                   negItems_clean,
                                                   batch_size=config["rec_model_p"]["batch_size"]))
            batch_users_clean, batch_pos_clean, batch_neg_clean = next(clean_data_iter)

        meta_model = deepcopy(recmodel)

        # ============= get input of the scheduler ============= #
        L_theta, _ = meta_model.loss(batch_users_clean, batch_pos_clean, batch_neg_clean, reduce=False)
        L_theta = torch.reshape(L_theta, (len(L_theta), 1))

        grads_theta_list = []
        for k in range(len(batch_users_clean)):
            grads_theta_list.append(torch.autograd.grad(L_theta[k], (meta_model.params()), create_graph=True,
                                              retain_graph=True))

        v_L_theta = ltw(L_theta.data)

        # assumed update
        L_theta_meta = torch.sum(L_theta * v_L_theta) / len(batch_users_clean)
        meta_model.zero_grad()
        grads = torch.autograd.grad(L_theta_meta, (meta_model.params()), create_graph=True, retain_graph=True)

        meta_model.update_params(lr_inner=config["rec_model_p"]["lr"], source_params=grads)
        del grads

        L_theta_hat, _ = meta_model.loss(batch_users_clean, batch_pos_clean, batch_neg_clean, reduce=False)

        # for each sample, calculate gradients of 2 losses
        input_embedding_cos = []
        for k in range(len(batch_users_clean)):
            task_grad_cos = []
            grads_theta = grads_theta_list[k]
            grads_theta_hat = torch.autograd.grad(L_theta_hat[k], (meta_model.params()), create_graph=True,
                                                  retain_graph=True)

            # calculate cosine similarity for each parameter
            for j in range(len(grads_theta)):
                task_grad_cos.append(scheduler.cosine(grads_theta[j].flatten().unsqueeze(0),
                                                      grads_theta_hat[j].flatten().unsqueeze(0))[0])
            del grads_theta
            del grads_theta_hat
            # stack similarity of each parameter
            task_grad_cos = torch.stack(task_grad_cos)
            # stack similarity of each sample
            input_embedding_cos.append(task_grad_cos.detach())

        # sample clean data
        weight = scheduler(L_theta, torch.stack(input_embedding_cos).cuda())

        task_prob = torch.softmax(weight.reshape(-1), dim=-1)
        log_p = torch.log(task_prob + 1e-20)
        logits = log_p.repeat([len(log_p), 1])

        sample_idx = scheduler.gumbel_softmax(logits, temperature=config["rec_model_p"]["tau"], hard=True)

        # ============= training ============= #
        meta_model = deepcopy(recmodel)

        # assumed update of theta (theta -> theta')
        cost, reg_loss = meta_model.loss(batch_users, batch_pos, batch_neg, reduce=False)
        cost_v = torch.reshape(cost, (len(cost), 1))
        v_lambda = ltw(cost_v.data)

        l_f_meta = torch.sum(cost_v * v_lambda) / len(batch_users)
        meta_model.zero_grad()
        grads = torch.autograd.grad(l_f_meta, (meta_model.params()), create_graph=True)

        # load theta' and update params of ltw
        meta_model.update_params(lr_inner=config["rec_model_p"]["lr"], source_params=grads)
        del grads

        if config["rec_model_p"]["model"] == 'lgn':
            user_emb, pos_emb, neg_emb, _, _, _ = meta_model.getEmbedding(batch_users_clean.long(),
                                                                        batch_pos_clean.long(), batch_neg_clean.long())
        else:
            user_emb, pos_emb, neg_emb = meta_model(batch_users_clean, batch_pos_clean, batch_neg_clean)

        batch_users_clean = torch.mm(sample_idx, user_emb)
        batch_pos_clean = torch.mm(sample_idx, pos_emb)
        batch_neg_clean = torch.mm(sample_idx, neg_emb)

        l_g_meta = meta_model.loss_gumbel(batch_users_clean, batch_pos_clean, batch_neg_clean)

        meta_opt.zero_grad()
        l_g_meta.backward(retain_graph=True)
        meta_opt.step()

        schedule_opt.zero_grad()
        l_g_meta.backward()
        schedule_opt.step()

        # reload and actually update theta
        cost_w, _ = recmodel.loss(batch_users, batch_pos, batch_neg, reduce=False)
        cost_v = torch.reshape(cost_w, (len(cost_w), 1))
        with torch.no_grad():
            w_new = ltw(cost_v)
        loss = torch.sum(cost_v * w_new) / len(batch_users)

        train_opt.zero_grad()
        loss.backward()
        train_opt.step()

        recmodel.store_params()

        train_loss += loss.cpu().item()
        meta_loss += l_g_meta.cpu().item()

    train_loss /= total_batch
    meta_loss /= total_batch
    timer.zero()
    return [f'{train_loss:.5f}', f'{meta_loss:.5f}']

def test_one_batch(X):
    sorted_items = X[0].numpy()
    groundTrue = X[1]
    r = rectool.getLabel(groundTrue, sorted_items)
    pre, recall, ndcg = [], [], []
    for k in config["topk"]:
        ret = rectool.RecallPrecision_ATk(groundTrue, r, k)
        pre.append(ret['precision'])
        recall.append(ret['recall'])
        ndcg.append(rectool.NDCGatK_r(groundTrue,r,k))
    return {'recall':np.array(recall),
            'precision':np.array(pre),
            'ndcg':np.array(ndcg)}

def test(config, logger, dataset, Recmodel, valid=True, multicore=0):
    u_batch_size = config["rec_model_p"]["test_u_batch_size"]
    dataset: Loader
    if valid:
        testDict = dataset.validDict
    else:
        testDict = dataset.testDict

    Recmodel = Recmodel.eval()
    max_K = max(config["topk"])

    if multicore == 1:
        pool = multiprocessing.Pool(CORES)
    results = {'precision': np.zeros(len(config["topk"])),
               'recall': np.zeros(len(config["topk"])),
               'ndcg': np.zeros(len(config["topk"]))}
    with torch.no_grad():
        users = list(testDict.keys())
        try:
            assert u_batch_size <= len(users) / 10
        except AssertionError:
            print(f"test_u_batch_size is too big for this dataset, try a small one {len(users) // 10}")

        users_list = []
        rating_list = []
        groundTrue_list = []
        total_batch = len(users) // u_batch_size + 1
        for batch_users in rectool.minibatch(users, batch_size=u_batch_size):
            allPos = dataset.getUserPosItems(batch_users)
            if not valid:
                validDict = dataset.validDict
                for i, user in enumerate(batch_users):
                    try:
                        allPos[i] = np.concatenate((allPos[i], validDict[user]))
                    except KeyError:
                        pass
            groundTrue = [testDict[u] for u in batch_users]
            batch_users_gpu = torch.Tensor(batch_users).long()
            batch_users_gpu = batch_users_gpu.to(config['device'])
            rating = Recmodel.getUsersRating(batch_users_gpu)
            exclude_index = []
            exclude_items = []
            for range_i, items in enumerate(allPos):
                exclude_index.extend([range_i] * len(items))
                exclude_items.extend(items)
            rating[exclude_index, exclude_items] = -(1 << 10)
            _, rating_K = torch.topk(rating, k=max_K)
            rating = rating.cpu().numpy()
            del rating
            users_list.append(batch_users)
            rating_list.append(rating_K.cpu())
            groundTrue_list.append(groundTrue)
        assert total_batch == len(users_list)
        X = zip(rating_list, groundTrue_list)
        if multicore == 1:
            pre_results = pool.map(test_one_batch, X)
        else:
            pre_results = []
            for x in X:
                pre_results.append(test_one_batch( x))
        for result in pre_results:
            results['recall'] += result['recall']
            results['precision'] += result['precision']
            results['ndcg'] += result['ndcg']
        results['recall'] /= float(len(users))
        results['precision'] /= float(len(users))
        results['ndcg'] /= float(len(users))
        if multicore == 1:
            pool.close()
        if not valid:
            logger.info(str(results))

    return results

def memorization_test(config, dataset, Recmodel):
    '''
    memorization procedure,
    update memorization history matrix and generate memorized data
    '''
    u_batch_size = config["rec_model_p"]["test_u_batch_size"]
    with torch.no_grad():
        users = dataset.trainUniqueUsers
        users_list = []
        items_list = []
        S = rectool.sample_K_neg(dataset)
        for batch_users in rectool.minibatch(users, batch_size=u_batch_size):
            allPos = dataset.getUserPosItems(batch_users)
            batch_users_gpu = torch.Tensor(batch_users).long()
            batch_users_gpu = batch_users_gpu.to(config['device'])

            rating = Recmodel.getUsersRating(batch_users_gpu)
            excluded_users = []
            excluded_items = []
            k_list = []
            for range_i, u in enumerate(batch_users):
                neg_items = S[u]
                items = allPos[range_i]
                k_list.append(len(items))
                neg_items.extend(items)
                excluded_items.extend(neg_items)
                excluded_users.extend([range_i] * (len(neg_items)))

            rating[excluded_users, excluded_items] += 100

            # rating_K: [batch_size, K]
            max_K = max(k_list)
            _, rating_K = torch.topk(rating, k=max_K)
            for i in range(len(rating_K)):
                user = batch_users[i]
                items = rating_K[i].tolist()[:k_list[i]]
                users_list.extend([user] * len(items))
                items_list.extend(items)
            try:
                assert len(users_list) == len(items_list)
            except AssertionError:
                print('len(users_list) != len(items_list)')
            del rating
        dataset.updateMemDict(users_list, items_list)
    return dataset.generate_clean_data()

