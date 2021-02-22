# -*- coding: utf-8 -*-
# @description: 
# @author: zchen
# @time: 2020/11/29 20:09
# @file: ner_main.py

from __future__ import absolute_import, division, print_function

import os
import torch
import torch.nn.functional as F
import pickle
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler, )

from tqdm import tqdm, trange
from tensorboardX import SummaryWriter

from pytorch_transformers import (BertConfig, BertTokenizer)
from pytorch_transformers import AdamW, WarmupLinearSchedule

from torch_ner.source.models import BERT_BiLSTM_CRF
from torch_ner.source.ner_processor import NerProcessor
from torch_ner.source.config import Config
import torch_ner.source.conlleval as conlleval
from torch_ner.source.logger import logger as logging


class NerMain(object):
    def __init__(self):
        self.config = Config()
        self.processor = NerProcessor()

    def train(self):
        """
        模型训练
        :return:
        """
        # 清理output_xxx目录，若output_xxx目录存在，将会被删除, 然后初始化输出目录
        self.processor.clean_output(self.config)

        # SummaryWriter构造函数
        writer = SummaryWriter(logdir=os.path.join(self.config.output_path, "eval"), comment="ner")

        # 配置可用设备，没有指定使用哪一块gpu，则全部使用
        device = torch.device('cuda' if torch.cuda.is_available() else self.config.device)
        self.config.device = device
        n_gpu = torch.cuda.device_count()
        logging.info(f"available device: {device}，count_gpu: {n_gpu}")

        # 如果显存不足，我们可以通过gradient_accumulation_steps梯度累计来解决
        if self.config.gradient_accumulation_steps < 1:
            raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                self.config.gradient_accumulation_steps))

        logging.info("====================== Start Data Pre-processing ======================")
        # 读取训练数据获取标签
        label_list = self.processor.get_labels(config=self.config)
        self.config.label_list = label_list
        num_labels = len(label_list)
        logging.info(f"loading labels successful! the size is {num_labels}, label is: {','.join(list(label_list))}")

        # 获取label2id、id2label的映射
        label2id, id2label = self.processor.get_label2id_id2label(self.config.output_path, label_list=label_list)
        logging.info("loading label2id and id2label dictionary successful!")

        if self.config.do_train:
            # 初始化tokenizer(分词器)、bert_config、bert_bilstm_crf model
            tokenizer = BertTokenizer.from_pretrained(self.config.model_name_or_path,
                                                      o_lower_case=self.config.do_lower_case)
            bert_config = BertConfig.from_pretrained(self.config.model_name_or_path, num_labels=num_labels)
            model = BERT_BiLSTM_CRF.from_pretrained(self.config.model_name_or_path, config=bert_config,
                                                    need_birnn=self.config.need_birnn, rnn_dim=self.config.rnn_dim)
            logging.info("loading tokenizer、bert_config and bert_bilstm_crf model successful!")

            model.to(device)
            if n_gpu > 1:
                model = torch.nn.DataParallel(model)

            # 获取训练样本、样本特征、TensorDataset信息
            train_examples, train_features, train_data = self.processor.get_dataset(self.config, tokenizer,
                                                                                    mode="train")
            # 训练数据载入
            train_data_loader = DataLoader(train_data, sampler=RandomSampler(train_data),
                                           batch_size=self.config.train_batch_size)
            logging.info("loading train data_set and data_loader successful!")

            eval_examples, eval_features, eval_data = [], [], None
            if self.config.do_eval:
                eval_examples, eval_features, eval_data = self.processor.get_dataset(self.config, tokenizer,
                                                                                     mode="eval")
                logging.info("loading eval data_set successful!")

            logging.info("====================== End Data Pre-processing ======================")

            # 计算优化器_模型参数的总更新次数、训练轮次
            if self.config.max_steps > 0:
                t_total = self.config.max_steps
                self.config.num_train_epochs = self.config.max_steps // (
                        len(train_data_loader) // self.config.gradient_accumulation_steps) + 1
            else:
                t_total = len(
                    train_data_loader) // self.config.gradient_accumulation_steps * self.config.num_train_epochs

            # 初始化模型优化器
            no_decay = ['bias', 'LayerNorm.weight']
            optimizer_grouped_parameters = [
                {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                 'weight_decay': 0.01},
                {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                 'weight_decay': 0.0}
            ]
            optimizer = AdamW(optimizer_grouped_parameters, lr=self.config.learning_rate, eps=self.config.adam_epsilon)
            scheduler = WarmupLinearSchedule(optimizer, warmup_steps=self.config.warmup_steps, t_total=t_total)
            logging.info("loading AdamW optimizer、Warmup LinearSchedule and calculate optimizer parameter successful!")

            logging.info("====================== Running training ======================")
            logging.info(f"Num Examples:  {len(train_data)}, Num Epochs: {self.config.num_train_epochs}，"
                         f"Num optimizer steps：{t_total}")

            # 启用 BatchNormalization 和 Dropout
            model.train()
            global_step, tr_loss, logging_loss, best_f1 = 0, 0.0, 0.0, 0.0
            for ep in trange(int(self.config.num_train_epochs), desc="Epoch"):
                model.train()
                for step, batch in enumerate(tqdm(train_data_loader, desc="DataLoader")):
                    batch = tuple(t.to(device) for t in batch)
                    input_ids, input_mask, segment_ids, label_ids = batch
                    outputs = model(input_ids, label_ids, segment_ids, input_mask)
                    loss = outputs

                    if n_gpu > 1:
                        loss = loss.mean()  # mean() to average on multi-gpu.

                    if self.config.gradient_accumulation_steps > 1:
                        loss = loss / self.config.gradient_accumulation_steps

                    loss.backward()
                    tr_loss += loss.item()

                    # 优化器_模型参数的总更新次数，和上面的t_total对应
                    if (step + 1) % self.config.gradient_accumulation_steps == 0:
                        optimizer.step()
                        scheduler.step()  # Update learning rate schedule
                        model.zero_grad()
                        global_step += 1

                        if self.config.logging_steps > 0 and global_step % self.config.logging_steps == 0:
                            tr_loss_avg = (tr_loss - logging_loss) / self.config.logging_steps
                            writer.add_scalar("Train/loss", tr_loss_avg, global_step)
                            logging_loss = tr_loss

                # 模型验证
                if self.config.do_eval:
                    logging.info("====================== Running Eval ======================")
                    all_ori_tokens_eval = [f.ori_tokens for f in eval_features]
                    overall, by_type = self.evaluate(self.config, eval_data, model, id2label, all_ori_tokens_eval)

                    # add eval result to tensorboard
                    f1_score = overall.fscore
                    writer.add_scalar("Eval/precision", overall.prec, ep)
                    writer.add_scalar("Eval/recall", overall.rec, ep)
                    writer.add_scalar("Eval/f1_score", overall.fscore, ep)

                    # save the best performs model
                    if f1_score > best_f1:
                        logging.info(f"***************** the best f1 is {f1_score}, save model *****************")
                        best_f1 = f1_score
                        # Take care of distributed/parallel training
                        model_to_save = model.module if hasattr(model,
                                                                'module') else model
                        model_to_save.save_pretrained(self.config.output_path)
                        tokenizer.save_pretrained(self.config.output_path)

                        # Good practice: save your training arguments together with the trained model
                        torch.save(self.config, os.path.join(self.config.output_path, 'training_args.bin'))
                        torch.save(model, os.path.join(self.config.output_path, 'ner_model.ckpt'))
                        logging.info("training_args.bin and ner_model.ckpt save successful!")
            writer.close()
            logging.info("bert_bilstm_crf model training successful!!!")

        if self.config.do_test:
            tokenizer = BertTokenizer.from_pretrained(self.config.output_path, do_lower_case=self.config.do_lower_case)
            config = torch.load(os.path.join(self.config.output_path, 'training_args.bin'))
            model = BERT_BiLSTM_CRF.from_pretrained(self.config.output_path, need_birnn=self.config.need_birnn,
                                                    rnn_dim=self.config.rnn_dim)
            model.to(device)

            test_examples, test_features, test_data = self.processor.get_dataset(config, tokenizer, mode="test")

            logging.info("====================== Running test ======================")
            logging.info(f"Num Examples:  {len(test_examples)}, Batch size: {config.eval_batch_size}")

            all_ori_tokens = [f.ori_tokens for f in test_features]
            all_ori_labels = [e.label.split(" ") for e in test_examples]
            test_sampler = SequentialSampler(test_data)
            test_data_loader = DataLoader(test_data, sampler=test_sampler, batch_size=config.eval_batch_size)
            model.eval()

            pred_labels = []

            for b_i, (input_ids, input_mask, segment_ids, label_ids) in enumerate(
                    tqdm(test_data_loader, desc="TestDataLoader")):

                input_ids = input_ids.to(device)
                input_mask = input_mask.to(device)
                segment_ids = segment_ids.to(device)

                with torch.no_grad():
                    logits = model.predict(input_ids, segment_ids, input_mask)

                for l in logits:
                    pred_label = []
                    for idx in l:
                        pred_label.append(id2label[idx])
                    pred_labels.append(pred_label)

            assert len(pred_labels) == len(all_ori_tokens) == len(all_ori_labels)

            with open(os.path.join(config.output_path, "token_labels_test.txt"), "w", encoding="utf-8") as f:
                for ori_tokens, ori_labels, prel in zip(all_ori_tokens, all_ori_labels, pred_labels):
                    for ot, ol, pl in zip(ori_tokens, ori_labels, prel):
                        if ot in ["[CLS]", "[SEP]"]:
                            continue
                        else:
                            f.write(f"{ot} {ol} {pl}\n")
                    f.write("\n")

    def predict(self, sentence):
        """
        模型预测
        :param sentence:
        :return:
        """
        max_seq_length = 128
        tokenizer = BertTokenizer.from_pretrained(self.config.output_path)
        text_list = list(sentence)
        tokens = []
        for word in text_list:
            tokens.extend(tokenizer.tokenize(word))

        if len(tokens) >= max_seq_length - 1:
            tokens = tokens[0:(max_seq_length - 2)]  # -2 的原因是因为序列需要加一个句首和句尾标志

        ntokens = ["[CLS]"] + tokens + ["[SEP]"]

        input_ids = tokenizer.convert_tokens_to_ids(ntokens)
        segment_ids = [0] * len(input_ids)
        input_mask = [1] * len(input_ids)

        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            segment_ids.append(0)
            input_mask.append(0)

        assert len(input_ids) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(input_mask) == max_seq_length

        input_ids = torch.tensor(input_ids, dtype=torch.long)
        segment_ids = torch.tensor(segment_ids, dtype=torch.long)
        input_mask = torch.tensor(input_mask, dtype=torch.long)

        input_ids = input_ids.to("cpu")
        segment_ids = segment_ids.to("cpu")
        input_mask = input_mask.to("cpu")

        input_ids = input_ids.unsqueeze(0)
        segment_ids = segment_ids.unsqueeze(0)
        input_mask = input_mask.unsqueeze(0)

        model = torch.load(os.path.join(self.config.output_path, "ner_model.ckpt"), map_location="cpu")

        if isinstance(model, torch.nn.DataParallel):
            model = model.module

        model.eval()
        with torch.no_grad():
            logits = model.predict(input_ids, segment_ids, input_mask)

        print(logits)
        with open(os.path.join(self.config.output_path, "label2id.pkl"), "rb") as f:
            label2id = pickle.load(f)

        id2label = {value: key for key, value in label2id.items()}

        pred_labels = []
        for l in logits:
            pred_label = []
            for idx in l:
                pred_label.append(id2label[idx])
            pred_labels.append(pred_label)

        print(pred_labels)

    @staticmethod
    def evaluate(config: Config, data, model, id2label, all_ori_tokens):
        ori_labels, pred_labels = [], []
        if isinstance(model, torch.nn.DataParallel):
            model = model.module
        model.eval()
        sampler = SequentialSampler(data)
        data_loader = DataLoader(data, sampler=sampler, batch_size=config.train_batch_size)
        for b_i, (input_ids, input_mask, segment_ids, label_ids) in enumerate(tqdm(data_loader, desc="Evaluating")):
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


if __name__ == '__main__':
    NerMain().predict("张三的爸爸是谁?")
