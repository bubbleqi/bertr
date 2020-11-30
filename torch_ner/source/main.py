# -*- coding: utf-8 -*-
# @description: 
# @author: zchen
# @time: 2020/11/29 20:09
# @file: main.py

from __future__ import absolute_import, division, print_function

import argparse
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


def main():
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    n_gpu = torch.cuda.device_count()
    args.device = device
    logging.info(f"device: {device}，n_gpu: {n_gpu}")

    # 如果显存不足，我们可以通过gradient_accumulation_steps梯度累计来解决
    if config.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
            config.gradient_accumulation_steps))

    # 清理output目录，若output目录存在，将会被删除
    if config.clean and config.do_train:
        logging.info(f"clear output dir: {config.output_path} ...")
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

    if os.path.exists(config.output_path) and os.listdir(config.output_path) and config.do_train:
        raise ValueError("Output directory ({}) already exists and is not empty.".format(config.output_path))

    if not os.path.exists(config.output_path):
        os.makedirs(config.output_path)

    if not os.path.exists(os.path.join(config.output_path, "eval")):
        os.makedirs(os.path.join(config.output_path, "eval"))

    writer = SummaryWriter(logdir=os.path.join(config.output_path, "eval"), comment="Linear")

    processor = NerProcessor()
    label_list = processor.get_labels(config=config)
    label_list_str = ",".join(list(label_list))
    num_labels = len(label_list)
    logging.info(f"labels size is {num_labels}, labels: {label_list_str}")
    args.label_list = label_list

    label2id, id2label = processor.get_label2id_id2label(config.output_path, label_list=label_list)

    if config.do_train:
        tokenizer = BertTokenizer.from_pretrained(
            config.model_name_or_path, do_lower_case=config.do_lower_case)

        bert_config = BertConfig.from_pretrained(
            config.model_name_or_path, num_labels=num_labels)

        model = BERT_BiLSTM_CRF.from_pretrained(
            config.model_name_or_path, config=bert_config,
            need_birnn=config.need_birnn, rnn_dim=config.rnn_dim)

        model.to(device)


if __name__ == '__main__':
    main()
