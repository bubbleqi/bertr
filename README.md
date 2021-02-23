#### BERT-BiLSTM-CRF模型

##### 【简介】使用谷歌的BERT模型在BiLSTM-CRF模型上进行预训练用于中文命名实体识别的pytorch代码

##### 项目结构
```
bert_bilstm_crf_ner_pytorch
    torch_ner
        bert-base-chinese           --- 预训练模型
        data                        --- 放置训练所需数据
        output                      --- 项目输出，包含模型、向量表示、日志信息等
        source                      --- 源代码
            config.py               --- 项目配置，模型参数
			models                  --- bert_bilstm_crf的torch实现
            conlleval.py            --- 模型验证
            ner_main.py             --- 训练主模块，包含训练、保存向量表示、预测等
            logger.py               --- 项目日志配置
            ner_processor.py        --- 数据预处理
            utils.py                --- 工具包
```
##### 数据预处理
输入数据格式请处理成BIO格式，放置在data/old_data目录下，格式如下:
```
在 O
广 B-LOC
西 I-LOC
壮 I-LOC
族 I-LOC
自 I-LOC
治 I-LOC
区 I-LOC
柳 I-LOC
州 I-LOC
市 I-LOC
柳 I-LOC
南 I-LOC
区 I-LOC
航 I-LOC
鹰 I-LOC
大 I-LOC
道 I-LOC
2 I-LOC
号 I-LOC
电 I-LOC
信 I-LOC
大 I-LOC
楼 I-LOC
租 O
房 O
住 O
7 B-DATE
年 I-DATE
的 O
田 B-FNAME
之 B-LNAME
桃 I-LNAME
是 O
什 O
么 O
人 O
？ O

在 O
广 B-LOC
西 I-LOC
壮 I-LOC
族 I-LOC
自 I-LOC
治 I-LOC
区 I-LOC
柳 I-LOC
州 I-LOC
市 I-LOC
柳 I-LOC
南 I-LOC
区 I-LOC
航 I-LOC
鹰 I-LOC
大 I-LOC
道 I-LOC
```
##### 运行环境
```
torch==1.1.0
pytorch_crf==0.7.2
numpy==1.19.4
pytorch_transformers==1.2.0
tqdm==4.51.0
PyYAML==5.3.1
tensorboardX==2.1
torchcrf==1.1.0
```

##### 使用方法
- 修改配置文件
- 训练
```
NerMain().train()
```
- 预测
```
NerMain().predict("xxx")
```

##### 参考文章
- [从Word Embedding到Bert模型——自然语言处理预训练技术发展史](https://mp.weixin.qq.com/s/FHDpx2cYYh9GZsa5nChi4g)
- [Pytorch-Bert预训练模型的使用](https://www.cnblogs.com/douzujun/p/13572694.html)
- [通俗易懂理解——BiLSTM](https://zhuanlan.zhihu.com/p/40119926)  
- [详解BiLSTM及代码实现](https://zhuanlan.zhihu.com/p/47802053)  
- [torch.nn.LSTM()详解](https://blog.csdn.net/m0_45478865/article/details/104455978)
- [LSTM细节分析理解（pytorch版）](https://zhuanlan.zhihu.com/p/79064602)
- [LSTM+CRF 解析（原理篇）](https://zhuanlan.zhihu.com/p/97829287)  
- [crf模型原理及解释](https://www.jianshu.com/p/e608cdfdc174)
