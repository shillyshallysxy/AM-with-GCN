import text_helper as th
import data_loader as dl
import tensorflow.compat.v1 as tf
import networkx as nx
import numpy as np
import os
from tqdm import tqdm
from HParameters import *
tf.disable_v2_behavior()


def write_binary(record_name, texts_, label_, label_p_, label_r_, label_d_, relation_graph_, node2pos_,
                 relation_graph_word_):
    record_path = os.path.join(ROOT_PATH, record_name)
    writer = tf.python_io.TFRecordWriter(record_path)
    max_mask_node = 0
    for it, text in tqdm(enumerate(texts_)):
        mask_ = len(text)*[1]
        node2pos_it_ = node2pos_[it]
        if len(node2pos_it_) > max_mask_node:
            max_mask_node = len(node2pos_it_)
        mask_node_ = len(node2pos_it_)*[1]
        relation_graph_it_ = get_relation_graph_vec(relation_graph_[it], node2pos_it_, MAX_LEN_NODE)
        node2pos_it_l_ = [temp_[0] for temp_ in node2pos_it_.values()]
        node2pos_it_r_ = [temp_[1] for temp_ in node2pos_it_.values()]
        example = tf.train.Example(
            features=tf.train.Features(
                feature={
                    "text": tf.train.Feature(int64_list=tf.train.Int64List(value=text)),
                    "mask": tf.train.Feature(int64_list=tf.train.Int64List(value=mask_)),
                    # "seg": tf.train.Feature(int64_list=tf.train.Int64List(value=seg_[it])),
                    "label_word": tf.train.Feature(int64_list=tf.train.Int64List(value=label_[it])),
                    "label_pos": tf.train.Feature(int64_list=tf.train.Int64List(value=label_p_[it])),
                    "label_relation": tf.train.Feature(int64_list=tf.train.Int64List(value=label_r_[it])),
                    "label_distance": tf.train.Feature(int64_list=tf.train.Int64List(value=label_d_[it])),
                    "mask_node": tf.train.Feature(int64_list=tf.train.Int64List(value=mask_node_)),
                    "relation_graph": tf.train.Feature(int64_list=tf.train.Int64List(value=relation_graph_it_)),
                    "node2pos_it_l": tf.train.Feature(int64_list=tf.train.Int64List(value=node2pos_it_l_)),
                    "node2pos_it_r": tf.train.Feature(int64_list=tf.train.Int64List(value=node2pos_it_r_)),
                    "relation_graph_word": tf.train.Feature(int64_list=tf.train.Int64List(value=relation_graph_word_[it])),
                }
            )
        )
        serialized = example.SerializeToString()
        writer.write(serialized)
    writer.close()
    logger("最长的node mask为：{}".format(max_mask_node))


def __parse_function(serial_exmp):
    features = tf.parse_single_example(serial_exmp, features={"text": tf.VarLenFeature(tf.int64),
                                                              "mask": tf.VarLenFeature(tf.int64),
                                                              # "seg": tf.VarLenFeature(tf.int64),
                                                              "label_word": tf.VarLenFeature(tf.int64),
                                                              "label_pos": tf.VarLenFeature(tf.int64),
                                                              "label_relation": tf.VarLenFeature(tf.int64),
                                                              "label_distance": tf.VarLenFeature(tf.int64),
                                                              "mask_node": tf.VarLenFeature(tf.int64),
                                                              "relation_graph": tf.VarLenFeature(tf.int64),
                                                              "node2pos_it_l": tf.VarLenFeature(tf.int64),
                                                              "node2pos_it_r": tf.VarLenFeature(tf.int64),
                                                              "relation_graph_word": tf.VarLenFeature(tf.int64),
                                                              })
    # text = tf.sparse_tensor_to_dense(features["text"], default_value=" ")
    texts_ = tf.sparse.to_dense(features["text"])
    mask_ = tf.sparse.to_dense(features["mask"])
    # seg_ = tf.sparse.to_dense(features["seg"])
    label_ = tf.sparse.to_dense(features["label_word"])
    label_p_ = tf.sparse.to_dense(features["label_pos"])
    label_r_ = tf.sparse.to_dense(features["label_relation"])
    label_d_ = tf.sparse.to_dense(features["label_distance"])

    mask_node_ = tf.sparse.to_dense(features["mask_node"])
    relation_graph_it_ = tf.sparse.to_dense(features["relation_graph"])
    node2pos_it_l_ = tf.sparse.to_dense(features["node2pos_it_l"])
    node2pos_it_r_ = tf.sparse.to_dense(features["node2pos_it_r"])
    relation_graph_word_ = tf.sparse.to_dense(features["relation_graph_word"])
    return texts_, mask_, label_, label_p_, label_r_, label_d_, mask_node_, \
           relation_graph_it_, node2pos_it_l_, node2pos_it_r_, relation_graph_word_


def get_dataset(record_name_):
    record_path_ = os.path.join(ROOT_PATH, record_name_)
    data_set_ = tf.data.TFRecordDataset(record_path_)
    return data_set_.map(__parse_function)


def get_relation_graph_vec(relation_graph_, node2pos_, max_length=MAX_LEN_NODE):
    relation_graph_it_ = nx.to_numpy_matrix(relation_graph_, nodelist=np.arange(len(node2pos_)), dtype=int)
    empty_graph = np.zeros(shape=(max_length, max_length), dtype=int)
    for ind in range(len(node2pos_)):
        for ind_ in range(len(node2pos_)):
            empty_graph[ind, ind_] = relation_graph_it_[ind, ind_]
    empty_graph = np.reshape(empty_graph, [-1])
    return empty_graph


def get_one_word_level_relation_graph(relation_graph_, node2pos_, max_length=560):
    relation_graph_ = nx.to_numpy_matrix(relation_graph_, nodelist=np.arange(len(node2pos_)))
    empty_graph = np.zeros(shape=(max_length, max_length), dtype=int)
    for ind, (l, r) in node2pos_.items():
        for ind_, (l_, r_) in node2pos_.items():
            empty_graph[l:r, l_:r_] = int(relation_graph_[ind, ind_])
    empty_graph = np.reshape(empty_graph, [-1])
    return empty_graph


def get_word_level_relation_graph(relation_graph_, node2pos_, max_length=560):
    res = list()
    for ind in range(len(relation_graph_)):
        res.append(get_one_word_level_relation_graph(relation_graph_[ind], node2pos_[ind], max_length=max_length))
    return res


if __name__ == "__main__":
    train_data, test_data = dl.load_essays(lower=LOWER, consider_other=CONSIDER_OTHER)
    train_texts, train_labels, train_labels_pos, train_labels_rel, train_labels_dis, \
    train_relation_graph, train_node2pos, _ = train_data
    train_relation_graph_word = get_word_level_relation_graph(train_relation_graph, train_node2pos)

    test_texts, test_labels, test_labels_pos, test_labels_rel, test_labels_dis, \
    test_relation_graph, test_node2pos, _ = test_data
    test_relation_graph_word = get_word_level_relation_graph(test_relation_graph, test_node2pos)

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

    write_binary(TRAIN_DATA_NAME, train_tokens, train_labels, train_labels_pos, train_labels_rel, train_labels_dis,
                 train_relation_graph, train_node2pos, train_relation_graph_word)
    write_binary(TEST_DATA_NAME, test_tokens, test_labels, test_labels_pos, test_labels_rel, test_labels_dis,
                 test_relation_graph, test_node2pos, test_relation_graph_word)

    if GLOVE:
        word_dict_ = dict()
        for k_, v_ in word_dict.vocab.items():
            word_dict_[k_] = v_.index
        word_dict = word_dict_

    import pickle
    with open(os.path.join(ROOT_PATH, WORD_DICT_NAME), 'wb') as f:
        pickle.dump(word_dict, f)
