import logging
import os

from torch_ner.source.config import Config

config = Config()

logger = logging.getLogger("client_log")
# Log等级总开关
logger.setLevel(logging.INFO)

# 创建handler，用于输出到控制台、写入日志文件
stream_handler = logging.StreamHandler()
log_file_handler = logging.FileHandler(filename=os.path.join(config.output_path, "bert_bilstm_crf_ner.log"),
                                       encoding="utf-8")

# 定义handler的输出格式
formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")

stream_handler.setFormatter(formatter)
log_file_handler.setFormatter(formatter)

logger.addHandler(stream_handler)
logger.addHandler(log_file_handler)