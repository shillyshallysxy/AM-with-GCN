# -*- encoding:utf8 -*-
import logging


def get_logger():
    logging.basicConfig(filename="./base.log",
                        format='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s',
                        filemode='a',
                        level=logging.INFO)

    logger = logging.getLogger()
    formatter = logging.Formatter('%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s')
    file_handler = logging.FileHandler("./base.log")
    file_handler.setFormatter(formatter)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger.info


BERT_CONFIG_PATH = "./bert_config.json"
MODEL_PATH = "./model/model.ckpt"

ROOT_PATH = "./temp"
TRAIN_DATA_NAME = "essayv2_train.tfrecord"
TEST_DATA_NAME = "essayv2_test.tfrecord"
WORD_DICT_NAME = "word_dict.pkl"
EMBEDDING_NAME = "glove_embedding.npy"
GLOVE = True

VECTOR_DIM = 200
GLOVE_NAME = "glove.6B.{}d.txt".format(VECTOR_DIM)

logger = get_logger()
# logger = print

