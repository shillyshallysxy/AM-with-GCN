import text_helper as th
import data_loader as dl
import tensorflow.compat.v1 as tf
import os
from tqdm import tqdm
from HParameters import *
tf.disable_v2_behavior()


def write_binary(record_name, texts_, label_):
    record_path = os.path.join(ROOT_PATH, record_name)
    writer = tf.python_io.TFRecordWriter(record_path)
    for it, text in tqdm(enumerate(texts_)):
        mask_ = len(text)*[1]
        example = tf.train.Example(
            features=tf.train.Features(
                feature={
                    "text": tf.train.Feature(int64_list=tf.train.Int64List(value=text)),
                    "mask": tf.train.Feature(int64_list=tf.train.Int64List(value=mask_)),
                    # "seg": tf.train.Feature(int64_list=tf.train.Int64List(value=seg_[it])),
                    "label": tf.train.Feature(int64_list=tf.train.Int64List(value=label_[it]))
                }
            )
        )
        serialized = example.SerializeToString()
        writer.write(serialized)
    writer.close()


def __parse_function(serial_exmp):
    features = tf.parse_single_example(serial_exmp, features={"text": tf.VarLenFeature(tf.int64),
                                                              "mask": tf.VarLenFeature(tf.int64),
                                                              # "seg": tf.VarLenFeature(tf.int64),
                                                              "label": tf.VarLenFeature(tf.int64)})
    # text = tf.sparse_tensor_to_dense(features["text"], default_value=" ")
    texts_ = tf.sparse.to_dense(features["text"])
    mask_ = tf.sparse.to_dense(features["mask"])
    # seg_ = tf.sparse.to_dense(features["seg"])
    label_ = tf.sparse.to_dense(features["label"])
    return texts_, mask_, label_


def get_dataset(record_name_):
    record_path_ = os.path.join(ROOT_PATH, record_name_)
    data_set_ = tf.data.TFRecordDataset(record_path_)
    return data_set_.map(__parse_function)


if __name__ == "__main__":
    train_data, test_data = dl.load_essays(lower=True)
    train_texts, train_labels = train_data
    test_texts, test_labels = test_data
    if GLOVE:
        logger("loading glove")
        word_dict = th.load_glove()
        import numpy as np
        np.save(os.path.join(ROOT_PATH, EMBEDDING_NAME), word_dict.vectors)
    else:
        word_dict = th.build_dictionary(train_texts)
    train_tokens = th.text_to_numbers(train_texts, word_dict, glove=GLOVE)
    test_tokens = th.text_to_numbers(test_texts, word_dict, glove=GLOVE)

    del train_texts, test_texts, train_data, test_data

    write_binary(TRAIN_DATA_NAME, train_tokens, train_labels)
    write_binary(TEST_DATA_NAME, test_tokens, test_labels)

