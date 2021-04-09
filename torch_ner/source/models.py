import torch.nn as nn
from torchcrf import CRF

from pytorch_transformers import BertPreTrainedModel, BertModel


class BERT_BiLSTM_CRF(BertPreTrainedModel):
    """
        BERT:
            outputs = self.bert(input_ids, token_type_ids=token_type_ids, attention_mask=input_mask)
            sequence_output = outputs[0]
            Args:
                input_ids:      torch.Size([batch_size,seq_len]), 代表输入实例的tensor张量
                token_type_ids: torch.Size([batch_size,seq_len]), 一个实例可以含有两个句子,相当于标记
                attention_mask: torch.Size([batch_size,seq_len]), 指定对哪些词进行self-Attention操作
            Out:
                sequence_output: torch.Size([batch_size,seq_len,hidden_size]), 输出序列
                pooled_output:   torch.Size([batch_size,hidden_size]), 对输出序列进行pool操作的结果
                (hidden_states): tuple, 13*torch.Size([batch_size,seq_len,hidden_size]), 隐藏层状态，取决于config的output_hidden_states
                (attentions):    tuple, 12*torch.Size([batch_size, 12, seq_len, seq_len]), 注意力层，取决于config中的output_attentions

        BiLSTM:
            self.birnn = nn.LSTM(config.hidden_size, rnn_dim, num_layers=1, bidirectional=True, batch_first=True)
            sequence_output, _ = self.birnn(sequence_output)
            Args:
                input_size:    输入数据的特征维数
                hidden_size:   LSTM中隐层的维度
                num_layers:    循环神经网络的层数
                bias:          用不用偏置，default=True
                batch_first:   通常我们输入的数据shape=(batch_size,seq_length,input_size),而batch_first默认是False,需要将batch_size与seq_length调换
                dropout:       默认是0，代表不用dropout
                bidirectional: 默认是false，代表不用双向LSTM
            Inputs:
                input:     shape=(seq_length,batch_size,input_size)的张量
                (h_0,c_0): h_0的shape=(num_layers*num_directions,batch,hidden_size)的张量，它包含了在当前这个batch_size中每个句子的初始隐藏状态，
                           num_layers就是LSTM的层数，如果bidirectional=True,num_directions=2,否则就是１，表示只有一个方向，c_0和h_0的形状相同，
                           它包含的是在当前这个batch_size中的每个句子的初始状态，h_0、c_0如果不提供，那么默认是０
            OutPuts:
                output:    shape=(seq_length,batch_size,num_directions*hidden_size), 它包含的LSTM的最后一层的输出特征(h_t),t是batch_size中每个句子的长度
                (h_n,c_n): h_n.shape=(num_directions * num_layers,batch,hidden_size), c_n与H_n形状相同, h_n包含的是句子的最后一个单词的隐藏状态，
                           c_n包含的是句子的最后一个单词的细胞状态，所以它们都与句子的长度seq_length无关;output[-1]与h_n是相等的，因为output[-1]包含的正是batch_size
                           个句子中每一个句子的最后一个单词的隐藏状态，注意LSTM中的隐藏状态其实就是输出，cell state细胞状态才是LSTM中一直隐藏的，记录着信息

        全连接层:
            self.hidden2tag = nn.Linear(out_dim, config.num_labels)
            Args:
                in_features:  输入的二维张量的大小，即输入的[batch_size, size]中的size
                out_features: 输出的二维张量的大小，即输出的二维张量的形状为[batch_size，output_size]，当然，它也代表了该全连接层的神经元个数

            释义:
                从输入输出的张量的shape角度来理解，相当于一个输入为[batch_size, in_features]的张量变换成了[batch_size, out_features]的输出张量

        CRF:
            self.crf = CRF(config.num_labels, batch_first=True)
            Args:
                num_tags:    Number of tags.
                batch_first: Whether the first dimension corresponds to the size of a minibatch.
    """

    def __init__(self, config, need_birnn=False, rnn_dim=128):
        super(BERT_BiLSTM_CRF, self).__init__(config)

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        out_dim = config.hidden_size
        self.need_birnn = need_birnn

        if need_birnn:
            self.birnn = nn.LSTM(config.hidden_size, rnn_dim, num_layers=1, bidirectional=True, batch_first=True)
            out_dim = rnn_dim * 2

        self.hidden2tag = nn.Linear(out_dim, config.num_labels)
        self.crf = CRF(config.num_labels, batch_first=True)

    def forward(self, input_ids, tags, token_type_ids=None, input_mask=None):
        emissions = self._tag_outputs(input_ids, token_type_ids, input_mask)
        loss = -1 * self.crf(emissions, tags, mask=input_mask.byte())
        return loss

    def _tag_outputs(self, input_ids, token_type_ids=None, input_mask=None):
        outputs = self.bert(input_ids, token_type_ids=token_type_ids, attention_mask=input_mask)
        sequence_output = outputs[0]
        if self.need_birnn:
            sequence_output, _ = self.birnn(sequence_output)
        sequence_output = self.dropout(sequence_output)
        emissions = self.hidden2tag(sequence_output)
        return emissions

    def predict(self, input_ids, token_type_ids=None, input_mask=None):
        emissions = self.tag_outputs(input_ids, token_type_ids, input_mask)
        return self.crf.decode(emissions, input_mask.byte())
