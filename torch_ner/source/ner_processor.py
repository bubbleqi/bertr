# -*- coding: utf-8 -*-
# @description: 
# @author: zchen
# @time: 2020/11/29 20:32
# @file: ner_processor.py
import logging
import os
import sys
import torch
import pickle

from torch.utils.data import TensorDataset
from torch_ner.source.config import Config
from tqdm import tqdm

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text, label=None):
        self.guid = guid
        self.text = text
        self.label = label


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id, ori_tokens):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.ori_tokens = ori_tokens


class NerProcessor(object):

    @staticmethod
    def get_labels(config: Config):
        """
        读取训练数据获取标签
        :param config:
        :return:
        """
        labels = set()
        if os.path.exists(os.path.join(config.output_path, "label_list.pkl")):
            logging.info(f"loading labels info from {config.output_path}")
            with open(os.path.join(config.output_path, "label_list.pkl"), "rb") as f:
                labels = pickle.load(f)
        else:
            # get labels from train data
            logging.info(f"loading labels info from train file and dump in {config.output_path}")
            with open(config.train_file, encoding="utf-8") as f:
                for line in f.readlines():
                    tokens = line.strip().split("\t")
                    if len(tokens) == 2:
                        labels.add(tokens[1])

            if len(labels) == 0:
                ValueError("loading labels error, labels type not found in data file: {}".format(config.output_path))

            with open(os.path.join(config.output_path, "label_list.pkl"), "wb") as f:
                pickle.dump(labels, f)
        return labels

    @staticmethod
    def get_label2id_id2label(output_path, label_list):
        """
        获取label2id、id2label的映射
        :param output_path:
        :param label_list:
        :return:
        """
        if os.path.exists(os.path.join(output_path, "label2id.pkl")):
            with open(os.path.join(output_path, "label2id.pkl"), "rb") as f:
                label2id = pickle.load(f)
        else:
            label2id = {l: i for i, l in enumerate(label_list)}
            with open(os.path.join(output_path, "label2id.pkl"), "wb") as f:
                pickle.dump(label2id, f)

        id2label = {value: key for key, value in label2id.items()}
        return label2id, id2label

    def get_dataset(self, config: Config, tokenizer, mode="train"):
        """
        获取数据集，包括InputExample(guid=guid, text=text, label=label)、features、TensorDataset
        :param config:
        :param tokenizer:
        :param mode:
        :return:
        """
        if mode == "train":
            filepath = config.train_file
        elif mode == "eval":
            filepath = config.eval_file
        elif mode == "test":
            filepath = config.test_file
        else:
            raise ValueError("mode must be one of train, eval, or test")

        # 通过读取输入数据，封装输入样本
        examples = self.get_input_examples(filepath)

        # 对输入数据进行特征转换
        features = self.convert_examples_to_features(config, examples, tokenizer)

        # 获取全部数据的特征，封装成TensorDataset
        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
        data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)

        return examples, features, data

    @staticmethod
    def convert_examples_to_features(config: Config, examples, tokenizer):
        """
        对输入数据进行特征转换
        :param config:
        :param examples:
        :param tokenizer:
        :return:
        """
        label_map = {label: i for i, label in enumerate(config.label_list)}
        max_seq_length = config.max_seq_length
        features = []
        for ex_index, example in tqdm(enumerate(examples), desc="convert examples"):
            example_text_list = example.text.split(" ")
            example_label_list = example.label.split(" ")
            assert len(example_text_list) == len(example_label_list)
            tokens, labels, ori_tokens = [], [], []
            for i, word in enumerate(example_text_list):
                # 防止wordPiece情况出现，不过貌似不会
                token = tokenizer.tokenize(word)
                tokens.extend(token)
                label_1 = example_label_list[i]
                ori_tokens.append(word)

                # 单个字符不会出现wordPiece
                for m in range(len(token)):
                    if m == 0:
                        labels.append(label_1)
                    else:
                        if label_1 == "O":
                            labels.append("O")
                        else:
                            labels.append("I")
            if len(tokens) >= max_seq_length - 1:
                # -2 的原因是因为序列需要加一个句首和句尾标志
                tokens = tokens[0:(max_seq_length - 2)]
                labels = labels[0:(max_seq_length - 2)]
                ori_tokens = ori_tokens[0:(max_seq_length - 2)]
            ori_tokens = ["[CLS]"] + ori_tokens + ["[SEP]"]

            ntokens, segment_ids, label_ids = [], [], []
            ntokens.append("[CLS]")
            segment_ids.append(0)
            label_ids.append(label_map["O"])

            for i, token in enumerate(tokens):
                ntokens.append(token)
                segment_ids.append(0)
                label_ids.append(label_map[labels[i]])

            ntokens.append("[SEP]")
            segment_ids.append(0)
            label_ids.append(label_map["O"])
            input_ids = tokenizer.convert_tokens_to_ids(ntokens)
            input_mask = [1] * len(input_ids)

            assert len(ori_tokens) == len(ntokens), f"{len(ori_tokens)}, {len(ntokens)}, {ori_tokens}"

            while len(input_ids) < max_seq_length:
                input_ids.append(0)
                input_mask.append(0)
                segment_ids.append(0)
                # we don't concerned about it!
                label_ids.append(0)
                ntokens.append("**NULL**")

            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length
            assert len(label_ids) == max_seq_length

            if ex_index < 5:
                logging.info("*** Example ***")
                logging.info("guid: %s" % example.guid)
                logging.info("tokens: %s" % " ".join([str(x) for x in ntokens]))
                logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
                logging.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
                logging.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
                logging.info("label_ids: %s" % " ".join([str(x) for x in label_ids]))

            features.append(InputFeatures(input_ids=input_ids,
                                          input_mask=input_mask,
                                          segment_ids=segment_ids,
                                          label_id=label_ids,
                                          ori_tokens=ori_tokens))
        return features

    def get_input_examples(self, input_file):
        """
        通过读取输入数据，封装输入样本
        :param input_file:
        :return:
        """
        examples = []
        lines = self.read_data(input_file)
        for i, line in enumerate(lines):
            guid = str(i)
            text = line[1]
            label = line[0]
            examples.append(InputExample(guid=guid, text=text, label=label))
        return examples

    @staticmethod
    def read_data(input_file):
        """
        读取输入数据
        :param input_file:
        :return:
        """
        with open(input_file, "r", encoding="utf-8") as f:
            lines, words, labels = [], [], []
            for line in f.readlines():
                contends = line.strip()
                tokens = line.strip().split("\t")
                if len(tokens) == 2:
                    words.append(tokens[0])
                    labels.append(tokens[1])
                else:
                    if len(contends) == 0 and len(words) > 0:
                        label = []
                        word = []
                        for l, w in zip(labels, words):
                            if len(l) > 0 and len(w) > 0:
                                label.append(l)
                                word.append(w)
                        lines.append([' '.join(label), ' '.join(word)])
                        words = []
                        labels = []
            return lines

    @staticmethod
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
