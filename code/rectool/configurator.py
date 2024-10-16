__all__ = ["Configurator"]

import os
import sys
import yaml
import json
import copy 


class Configurator(object):

    def __init__(self):
        """Initializes a new `Configurator` instance.
        """
        self._cnf = {}
        self._cmd_args = {}
    def __deepcopy__(self, memo):
        # 创建一个新的实例
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result

        # 深拷贝所有属性
        for k, v in self.__dict__.items():
            setattr(result, k, copy.deepcopy(v, memo))
        return result
    def __getstate__(self):
        state = self.__dict__.copy()
        # 你可以在这里移除不需要序列化的属性
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

    
    def add_config(self, cfg_file,):
        """Read and add config from yaml-style file.
        支持加载多个文件，存在冲突时，后面的文件会覆盖前面的文件
        """
        if not os.path.isfile(cfg_file):
            raise FileNotFoundError("File '%s' does not exist." % cfg_file)

        with open(cfg_file, 'r') as f:
            config = yaml.safe_load(f)
        self._cnf.update(config)
    
    def extract_config(self, cfg_file, name_key, new_key ):
        """
        根据name_key的value，从cfg_file抽取配置信息添加到new_key中；
        """
        value = self._cnf[name_key]

        with open(cfg_file, 'r') as f:
            new_config = yaml.safe_load(f)

        if value in new_config and new_config[value]!=None:
            self._cnf.update({new_key:new_config[value]})

        
        # self._cnf[new_key] = new_config[value]
        



        

    def __getitem__(self, item):
        param = self._cnf[item]
        return param

    def __getattr__(self, item):
        return self[item]
    
    def __setitem__(self, key, value):
        self._cnf[key] = value

    def __contains__(self, o):
        if o in self._cnf:
            flag = True
        else:
            flag = False
        

        # for sec_name, sec_args in self._cnf.items():
        #     if o in sec_args:
        #         flag = True
        #         break
        # else:
        #     if o in self._cmd_args:
        #         flag = True
        #     else:
        #         flag = False

        return flag


    def __str__(self):
        """
        Returns:
            str: yaml-style string of the configurator's arguments.
        """
        str_cnf = yaml.dump(self._cnf, default_flow_style=False)
        return str_cnf

    def __repr__(self):
        return self.__str__()

