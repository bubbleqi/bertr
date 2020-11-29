# -*- coding: utf-8 -*-
# @description: 
# @author: zchen
# @time: 2020/11/29 19:49
# @file: config.py
import os
import threading


class Config(object):
    _instance_lock = threading.Lock()
    __init_flag = False

    def __init__(self):
        if not Config.__init_flag:
            print("===================>>>Config<<<=========================")
            Config.__init_flag = True
            root_path = str(os.getcwd()).replace("\\", "/")
            if 'source' in root_path.split('/'):
                self.base_path = os.path.join(os.path.pardir)
            else:
                self.base_path = os.path.join(os.getcwd(), 'torch_ner')

    def __new__(cls, *args, **kwargs):
        """
        单例类
        :param args:
        :param kwargs:
        :return:
        """
        if not hasattr(Config, '_instance'):
            with Config._instance_lock:
                if not hasattr(Config, '_instance'):
                    Config._instance = object.__new__(cls)
        return Config._instance

    def _init_train_config(self):
        self.train_file = os.path.join(self.base_path, 'data', 'train.txt')
        self.eval_file = os.path.join(self.base_path, 'data', 'eval.txt')
        self.test_file = os.path.join(self.base_path, 'data', 'test.txt')
