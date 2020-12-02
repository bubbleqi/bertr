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
import torch_ner.source.conlleval as conlleval

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
config = Config()


# set the random seed for repeat
def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def to_list(tensor):
    return tensor.detach().cpu().tolist()


def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'


def evaluate(config: Config, data, model, id2label, all_ori_tokens):
    ori_labels, pred_labels = [], []
    model.eval()
    sampler = SequentialSampler(data)
    dataloader = DataLoader(data, sampler=sampler, batch_size=config.train_batch_size)
    for b_i, (input_ids, input_mask, segment_ids, label_ids) in enumerate(tqdm(dataloader, desc="Evaluating")):
        input_ids = input_ids.to(config.device)
        input_mask = input_mask.to(config.device)
        segment_ids = segment_ids.to(config.device)
        label_ids = label_ids.to(config.device)
        with torch.no_grad():
            logits = model.predict(input_ids, segment_ids, input_mask)

        for l in logits:
            pred_labels.append([id2label[idx] for idx in l])

        for l in label_ids:
            ori_labels.append([id2label[idx.item()] for idx in l])

    eval_list = []
    for ori_tokens, oril, prel in zip(all_ori_tokens, ori_labels, pred_labels):
        for ot, ol, pl in zip(ori_tokens, oril, prel):
            if ot in ["[CLS]", "[SEP]"]:
                continue
            eval_list.append(f"{ot} {ol} {pl}\n")
        eval_list.append("\n")

    # eval the model
    counts = conlleval.evaluate(eval_list)
    conlleval.report(counts)

    # namedtuple('Metrics', 'tp fp fn prec rec fscore')
    overall, by_type = conlleval.metrics(counts)

    return overall, by_type


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
    device = torch.device('cuda' if torch.cuda.is_available() else config.device)
    config.device = device
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

        eval_examples, eval_features, eval_data = [], [], None
        if config.do_eval:
            eval_examples, eval_features, eval_data = processor.get_dataset(config, tokenizer, mode="eval")

        logging.info("loading AdamW optimizer、WarmupLinearSchedule and calculate optimizer parameter...")
        # 计算优化器_模型参数的总更新次数、训练轮次
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

        logging.info("************** Running training ****************")
        logging.info("Num examples = %d", len(train_data))
        logging.info("Num Epochs = %d", config.num_train_epochs)
        logging.info("Total optimization steps = %d", t_total)

        # 启用 BatchNormalization 和 Dropout
        model.train()
        global_step, tr_loss, logging_loss, best_f1 = 0, 0.0, 0.0, 0.0
        for ep in trange(int(config.num_train_epochs), desc="Epoch"):
            model.train()
            for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
                batch = tuple(t.to(device) for t in batch)
                input_ids, input_mask, segment_ids, label_ids = batch
                outputs = model(input_ids, label_ids, segment_ids, input_mask)
                loss = outputs

                if n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu.

                if config.gradient_accumulation_steps > 1:
                    loss = loss / config.gradient_accumulation_steps

                loss.backward()
                tr_loss += loss.item()

                # 优化器_模型参数的总更新次数，和上面的t_total对应
                if (step + 1) % config.gradient_accumulation_steps == 0:
                    optimizer.step()
                    scheduler.step()  # Update learning rate schedule
                    model.zero_grad()
                    global_step += 1

                    if config.logging_steps > 0 and global_step % config.logging_steps == 0:
                        tr_loss_avg = (tr_loss - logging_loss) / config.logging_steps
                        writer.add_scalar("Train/loss", tr_loss_avg, global_step)
                        logging_loss = tr_loss

            # 模型验证
            if config.do_eval:
                logging.info("********* Running eval **********")
                all_ori_tokens_eval = [f.ori_tokens for f in eval_features]
                overall, by_type = evaluate(config, eval_data, model, id2label, all_ori_tokens_eval)

                # add eval result to tensorboard
                f1_score = overall.fscore
                writer.add_scalar("Eval/precision", overall.prec, ep)
                writer.add_scalar("Eval/recall", overall.rec, ep)
                writer.add_scalar("Eval/f1_score", overall.fscore, ep)

                # save the best performs model
                if f1_score > best_f1:
                    logging.info(f"----------the best f1 is {f1_score}, save model---------")
                    best_f1 = f1_score
                    # Take care of distributed/parallel training
                    model_to_save = model.module if hasattr(model,
                                                            'module') else model
                    model_to_save.save_pretrained(config.output_path)
                    tokenizer.save_pretrained(config.output_path)

                    # Good practice: save your training arguments together with the trained model
                    torch.save(config, os.path.join(config.output_path, 'training_args.bin'))
                    torch.save(model, os.path.join(config.output_path, 'ner_model.ckpt'))
        writer.close()
        logging.info("bert_bilstm_crf model training successful!!!")


if __name__ == '__main__':
    train()
