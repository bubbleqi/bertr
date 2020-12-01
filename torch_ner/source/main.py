# -*- coding: utf-8 -*-
# @description: 
# @author: zchen
# @time: 2020/11/29 20:09
# @file: main.py

from __future__ import absolute_import, division, print_function

import csv
import logging
import os
import random
import json
import sys
import datetime
import time
import numpy as np
import torch
import torch.nn.functional as F
import pickle
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler

from tqdm import tqdm, trange
from tensorboardX import SummaryWriter

from pytorch_transformers import (WEIGHTS_NAME, BertConfig, BertTokenizer)
from pytorch_transformers import AdamW, WarmupLinearSchedule

from torch_ner.source.models import BERT_BiLSTM_CRF
from torch_ner.source.ner_processor import NerProcessor
from torch_ner.source.config import Config

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
config = Config()


def clean_output(config: Config):
    """
    清理output目录，若output目录存在，将会被删除, 然后初始化输出目录
    :param config:
    :return:
    """
    if config.clean and config.do_train:
        logging.info(f"clear output dir: {config.output_path}")
        if os.path.exists(config.output_path):
            def del_file(path):
                ls = os.listdir(path)
                for i in ls:
                    c_path = os.path.join(path, i)
                    if os.path.isdir(c_path):
                        del_file(c_path)
                        os.rmdir(c_path)
                    else:
                        os.remove(c_path)

            try:
                del_file(config.output_path)
            except Exception as e:
                logging.error(e)
                logging.error('pleace remove the files of output dir and data.conf')
                exit(-1)

    # 初始化output目录
    if os.path.exists(config.output_path) and os.listdir(config.output_path) and config.do_train:
        raise ValueError("Output directory ({}) already exists and is not empty.".format(config.output_path))

    if not os.path.exists(config.output_path):
        os.makedirs(config.output_path)

    if not os.path.exists(os.path.join(config.output_path, "eval")):
        os.makedirs(os.path.join(config.output_path, "eval"))


def train():
    # 配置可用设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    n_gpu = torch.cuda.device_count()
    logging.info(f"available device: {device}，count_gpu: {n_gpu}")

    # 如果显存不足，我们可以通过gradient_accumulation_steps梯度累计来解决
    if config.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
            config.gradient_accumulation_steps))

    # 清理output目录，若output目录存在，将会被删除, 然后初始化输出目录
    clean_output(config)

    writer = SummaryWriter(logdir=os.path.join(config.output_path, "eval"), comment="Linear")

    # ===============数据预处理================
    processor = NerProcessor()
    logging.info("now starting data pre-processing...")

    # 读取训练数据获取标签
    label_list = processor.get_labels(config=config)
    config.label_list = label_list
    num_labels = len(label_list)
    logging.info(f"labels size is {num_labels}, labels: {','.join(list(label_list))}")

    # 获取label2id、id2label的映射
    label2id, id2label = processor.get_label2id_id2label(config.output_path, label_list=label_list)
    logging.info("initialize label2id/id2label dictionary successful")

    if config.do_train:
        # 初始化tokenizer(分词器)、bert_config、bert_bilstm_crf model
        tokenizer = BertTokenizer.from_pretrained(config.model_name_or_path, do_lower_case=config.do_lower_case)
        bert_config = BertConfig.from_pretrained(config.model_name_or_path, num_labels=num_labels)
        model = BERT_BiLSTM_CRF.from_pretrained(config.model_name_or_path, config=bert_config,
                                                need_birnn=config.need_birnn, rnn_dim=config.rnn_dim)
        logging.info("building tokenizer、bert_config and bert_bilstm_crf model successful")
        model.to(device)

        if n_gpu > 1:
            model = torch.nn.DataParallel(model)

        logging.info("loading train data and dataloader...")
        # 获取训练样本、样本特征、TensorDataset信息
        train_examples, train_features, train_data = processor.get_dataset(config, tokenizer, mode="train")
        # 初始化RandomSampler采样器
        train_sampler = RandomSampler(train_data)
        # 数据加载器，结合了数据集和取样器，并且可以提供多个线程处理数据集
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=config.train_batch_size)

        if config.do_eval:
            eval_examples, eval_features, eval_data = processor.get_dataset(config, tokenizer, mode="train")

        logging.info("loading AdamW optimizer、WarmupLinearSchedule and calculate optimizer parameter...")
        # 计算优化器更新次数、训练轮次
        if config.max_steps > 0:
            t_total = config.max_steps
            config.num_train_epochs = config.max_steps // (
                    len(train_dataloader) // config.gradient_accumulation_steps) + 1
        else:
            t_total = len(train_dataloader) // config.gradient_accumulation_steps * config.num_train_epochs

        # 初始化模型优化器
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': 0.01},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=config.learning_rate, eps=config.adam_epsilon)
        scheduler = WarmupLinearSchedule(optimizer, warmup_steps=config.warmup_steps, t_total=t_total)

        logging.info("*********** Running training ***********")
        logging.info("Num examples = %d", len(train_data))
        logging.info("Num Epochs = %d", config.num_train_epochs)
        logging.info("Total optimization steps = %d", t_total)


if __name__ == '__main__':
    train()
