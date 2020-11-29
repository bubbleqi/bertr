# -*- coding: utf-8 -*-
# @description: 
# @author: zchen
# @time: 2020/11/29 20:32
# @file: ner_processor.py
import logging
import os
import pickle

from torch_ner.source.config import Config

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


class NerProcessor(object):

    @staticmethod
    def get_labels(config: Config):
        labels = set()
        if os.path.exists(os.path.join(config.output_path, "label_list.pkl")):
            logger.info(f"loading labels info from {config.output_path}")
            with open(os.path.join(config.output_path, "label_list.pkl"), "rb") as f:
                labels = pickle.load(f)
        else:
            # get labels from train data
            logger.info(f"loading labels info from train file and dump in {config.output_path}")
            with open(config.train_file, encoding="utf-8") as f:
                for line in f.readlines():
                    tokens = line.strip().split("\t")
                    if len(tokens) == 2:
                        labels.add(tokens[1])

            if len(labels) == 0:
                logger.info("loading error and return the default labels B,I,O")
                labels = {"O", "B", "I"}

            with open(os.path.join(config.output_path, "label_list.pkl"), "wb") as f:
                pickle.dump(labels, f)

        return labels

    @staticmethod
    def get_label2id_id2label(config, label_list):
        if os.path.exists(os.path.join(config.output_path, "label2id.pkl")):
            with open(os.path.join(config.output_path, "label2id.pkl"), "rb") as f:
                label2id = pickle.load(f)
        else:
            label2id = {l: i for i, l in enumerate(label_list)}
            with open(os.path.join(config.output_path, "label2id.pkl"), "wb") as f:
                pickle.dump(label2id, f)

        id2label = {value: key for key, value in label2id.items()}
        return label2id, id2label

    def read_data(self, input_file):
        """Reads a BIO data."""
        with open(input_file, "r", encoding="utf-8") as f:
            lines = []
            words = []
            labels = []

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
