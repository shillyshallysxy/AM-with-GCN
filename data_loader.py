# -*- coding: utf-8 -*-
import csv
import os
import string
from collections import Counter
from scipy.sparse import csr_matrix
import networkx as nx
import nltk
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from HParameters import *

tokenizer = nltk.word_tokenize

entities = {"PAD": 0, "MajorClaim": 1, "Claim": 2, "Premise": 3, "Other": 0}
pos = {"PAD": 0, "Begin": 1, "Intermediate": 2, "End": 3, "Single": 4, "Other": 0}
relations = {"PAD": 0, "supports": 1, "attacks": 2, "For": 3, "Against": 4, "Other": 0}
attributes = {"PAD": 0, "Stance": 1, "Other": 0}
trans_table = {ord(f): ord(t) for f, t in zip(
     u'，。！？【】（）％＃＠＆１２３４５６７８９０‘’“”""',
     u',.!?[]()%#@&1234567890\'\'\'\'\'\'')}


def load_elecdeb60to16(data_path='data/stackoverflow/'):

    # load SO embedding
    with open(data_path + 'vocab_withIdx.dic', 'r', encoding="utf-8") as inp_indx, \
            open(data_path + 'vocab_emb_Word2vec_48_index.dic', 'r', encoding="utf-8") as inp_dic, \
            open(data_path + 'vocab_emb_Word2vec_48.vec', encoding="utf-8") as inp_vec:
        pair_dic = inp_indx.readlines()
        word_index = {}
        for pair in pair_dic:
            word, index = pair.replace('\n', '').split('\t')
            word_index[word] = index

        index_word = {v: k for k, v in word_index.items()}

        del pair_dic

        emb_index = inp_dic.readlines()
        emb_vec = inp_vec.readlines()
        word_vectors = {}
        for index, vec in zip(emb_index, emb_vec):
            word = index_word[index.replace('\n', '')]
            word_vectors[word] = np.array(list((map(float, vec.split()))))

        del emb_index
        del emb_vec

    with open(data_path + 'title_StackOverflow.txt', 'r', encoding="utf-8") as inp_txt:
        all_lines = inp_txt.readlines()[:-1]
        text_file = " ".join([" ".join(nltk.word_tokenize(c)) for c in all_lines])
        word_count = Counter(text_file.split())
        total_count = sum(word_count.values())
        unigram = {}
        for item in word_count.items():
            unigram[item[0]] = item[1] / total_count

        all_vector_representation = np.zeros(shape=(20000, 48))
        for i, line in enumerate(all_lines):
            word_sentence = nltk.word_tokenize(line)

            sent_rep = np.zeros(shape=[48, ])
            j = 0
            for word in word_sentence:
                try:
                    wv = word_vectors[word]
                    j = j + 1
                except KeyError:
                    continue

                weight = 0.1 / (0.1 + unigram[word])
                sent_rep += wv * weight
            if j != 0:
                all_vector_representation[i] = sent_rep / j
            else:
                all_vector_representation[i] = sent_rep

    pca = PCA(n_components=1)
    pca.fit(all_vector_representation)
    pca = pca.components_

    temp1 = all_vector_representation.dot(pca.transpose())
    temp2 = temp1 * pca
    temp3 = all_vector_representation - temp2
    XX1 = all_vector_representation - all_vector_representation.dot(pca.transpose()) * pca
    # 1 2 3 0.5 0.2 0.3 0.5 0.4 0.9 1.8* 0.9 0.36 0.54
    XX = XX1

    scaler = MinMaxScaler()
    XX = scaler.fit_transform(XX)

    with open(data_path + 'label_StackOverflow.txt') as label_file:
        y = np.array(list((map(int, label_file.readlines()))))
        print(y.dtype)

    return XX, y


def load_essays(data_path="./data/ArgumentAnnotatedEssays-2.0", lower=False):
    detail_data_path = os.path.join(data_path, "brat-project-final")
    # 读取训练测试集分割
    train_test_split_path = os.path.join(data_path, "train-test-split.csv")
    train_test_split = dict()
    with open(train_test_split_path, 'r', encoding="utf-8") as f_:
        reader_ = csv.reader(f_)
        next(reader_)
        for row_ in reader_:
            row_ = row_[0].lower()
            row_ = row_.replace("'", "")
            row_ = row_.replace("\"", "")
            row_ = row_.split(";")
            try:
                train_test_split[row_[1]].append(row_[0])
            except KeyError:
                train_test_split[row_[1]] = list()
                train_test_split[row_[1]].append(row_[0])
    data_list = os.listdir(detail_data_path)
    accept_types = ["txt", "ann", ]
    data_dict = dict()
    logger("loading data path...")
    for data_ in data_list:
        if data_.startswith("essay"):
            essay_name_, _, essay_type_ = data_.rpartition('.')
            if essay_type_ in accept_types:
                try:
                    data_dict[essay_name_][essay_type_] = data_
                except KeyError:
                    data_dict[essay_name_] = dict()
                    data_dict[essay_name_][essay_type_] = data_
            else:
                logger("during loading data path: unknown type: {}".format(data_))
    logger("loading data content...")
    for k_, v_ in data_dict.items():
        for accept_type_ in accept_types:
            data_name_ = v_[accept_type_]
            with open(os.path.join(detail_data_path, data_name_), 'r', encoding="utf-8") as f_:
                data_content_ = f_.readlines()
                if accept_type_ == "txt":
                    data_dict[k_][accept_type_] = parse_txt_content(data_content_, lower=lower)
                elif accept_type_ == "ann":
                    data_dict[k_][accept_type_] = parse_ann_content(data_content_, lower=lower)
                else:
                    logger("during loading data content: unknown type: {}".format(accept_type_))
    # -------做单词级别的序列标注---------
    for k_, v_ in data_dict.items():
        data_dict[k_]["entities_label_char"] = np.ones(len(v_["txt"][2]), dtype=np.int)*entities["Other"]
        data_dict[k_]["node2pos_label_char"] = np.ones(len(v_["txt"][2]), dtype=np.int)*entities["Other"]
        data_dict[k_]["entities_label_word"] = np.ones(len(v_["txt"][0]), dtype=np.int)*entities["Other"]
        data_dict[k_]["node2pos_label_word"] = np.ones(len(v_["txt"][0]), dtype=np.int)*entities["Other"]
        data_dict[k_]["entities_label_pos"] = np.ones(len(v_["txt"][0]), dtype=np.int)*entities["Other"]
        # 做字符集别的序列标注
        for a_k_, a_v_ in v_["ann"].items():
            if a_v_["type"] in entities:
                end = int(a_v_["content"][1])
                start = int(a_v_["content"][0])
                debug_ = v_["txt"][2][start:end]
                assert debug_ == a_v_["article"]
                data_dict[k_]["entities_label_char"][start: end] = entities[a_v_["type"]]
                data_dict[k_]["node2pos_label_char"][start: end] = int(a_k_.replace("T", ""))
        # 做单词级别的序列标注
        now_ind_ = 0
        for ind_, word_ in enumerate(v_["txt"][0]):
            # 一些规则，但鲁棒性不行
            if word_[0] in string.punctuation:
                now_ind_ -= 1
            if word_ == "n\'t":
                now_ind_ -= 1

            while v_["txt"][2][now_ind_] in string.whitespace:
                now_ind_ += 1

            label_ = data_dict[k_]["entities_label_char"][now_ind_: now_ind_ + len(word_)]
            debug_ = v_["txt"][2][now_ind_: now_ind_ + len(word_)]
            # 自适应调整匹配
            if debug_ != word_:
                search_range = 2
                bias = -search_range
                while debug_ != word_ and bias <= search_range:
                    debug_ = v_["txt"][2][now_ind_+bias: now_ind_+bias + len(word_)]
                    bias += 1
                now_ind_ += (bias-1)

            assert debug_ == word_
            label_ = sorted(label_)[len(label_) // 2]
            data_dict[k_]["entities_label_word"][ind_] = label_

            label_ = data_dict[k_]["node2pos_label_char"][now_ind_: now_ind_ + len(word_)]
            label_ = sorted(label_)[len(label_) // 2]
            data_dict[k_]["node2pos_label_word"][ind_] = label_

            now_ind_ += (len(word_)+1)

        del data_dict[k_]["entities_label_char"], data_dict[k_]["node2pos_label_char"]
        # POS
        entities_labels_ = data_dict[k_]["entities_label_word"]
        if entities_labels_[0] == entities["Other"]:
            data_dict[k_]["entities_label_pos"][0] = pos["Other"]
        else:
            if entities_labels_[0] != entities_labels_[1]:
                data_dict[k_]["entities_label_pos"][0] = pos["Single"]
            else:
                data_dict[k_]["entities_label_pos"][0] = pos["Begin"]
        for ind_ in range(1, len(entities_labels_)-1):
            if entities_labels_[ind_] == entities["Other"]:
                data_dict[k_]["entities_label_pos"][ind_] = pos["Other"]
            else:
                if entities_labels_[ind_-1] != entities_labels_[ind_]:
                    if entities_labels_[ind_] != entities_labels_[ind_+1]:
                        data_dict[k_]["entities_label_pos"][ind_] = pos["Single"]
                    else:
                        data_dict[k_]["entities_label_pos"][ind_] = pos["Begin"]
                else:
                    if entities_labels_[ind_] != entities_labels_[ind_+1]:
                        data_dict[k_]["entities_label_pos"][ind_] = pos["End"]
                    else:
                        data_dict[k_]["entities_label_pos"][ind_] = pos["Intermediate"]
        if entities_labels_[-1] == entities["Other"]:
            data_dict[k_]["entities_label_pos"][-1] = pos["Other"]
        else:
            if entities_labels_[-1] != entities_labels_[-2]:
                data_dict[k_]["entities_label_pos"][-1] = pos["Single"]
            else:
                data_dict[k_]["entities_label_pos"][-1] = pos["End"]

        # 构建节点对应文本位置
        prior_ind_ = 0
        data_dict[k_]["node2pos"] = dict()
        temp_trans_map = dict()
        for ind_ in range(len(data_dict[k_]["entities_label_word"]) - 1):
            if data_dict[k_]["entities_label_word"][ind_] != data_dict[k_]["entities_label_word"][ind_ + 1]:
                node2pos_key_ = len(data_dict[k_]["node2pos"])
                data_dict[k_]["node2pos"][node2pos_key_] = (prior_ind_, ind_ + 1)

                label_ = data_dict[k_]["node2pos_label_word"][prior_ind_: ind_ + 1]
                label_ = sorted(label_)[len(label_) // 2]
                temp_trans_map[label_] = node2pos_key_

                prior_ind_ = ind_ + 1

        # 做关系标注
        data_dict[k_]["relation_graph"] = dict()
        temp_col_ = list()
        temp_row_ = list()
        temp_data_ = list()
        temp_major_inds_ = list()
        for a_k_, a_v_ in v_["ann"].items():
            if a_v_["type"] in relations:
                from_ = int(a_v_["content"][0].replace("T", ""))
                to_ = int(a_v_["content"][1].replace("T", ""))
                from_ = temp_trans_map[from_]
                to_ = temp_trans_map[to_]
                relation_code_ = relations[a_v_["type"]]
                if from_ not in data_dict[k_]["relation_graph"]:
                    data_dict[k_]["relation_graph"][from_] = list()
                data_dict[k_]["relation_graph"][from_].append(to_)
                temp_row_.append(from_)
                temp_col_.append(to_)
                temp_data_.append(relation_code_)
            elif a_v_["type"] == "MajorClaim":
                temp_major_inds_.append(temp_trans_map[int(a_k_.replace("T", ""))])
        for a_k_, a_v_ in v_["ann"].items():
            if a_v_["type"] in attributes:
                from_ = int(a_v_["content"][0].replace("T", ""))
                from_ = temp_trans_map[from_]
                relation_code_ = relations[a_v_["content"][1]]
                for temp_major_ind_ in temp_major_inds_:
                    temp_row_.append(from_)
                    temp_col_.append(temp_major_ind_)
                    temp_data_.append(relation_code_)
        temp_shape_ = len(data_dict[k_]["node2pos"])
        data_dict[k_]["adj_graph"] = csr_matrix((temp_data_, (temp_row_, temp_col_)),
                                                shape=(temp_shape_, temp_shape_))
        data_dict[k_]["relation_graph"] = nx.from_dict_of_lists(data_dict[k_]["relation_graph"])

        # data_dict[k_]["adj_graph"] = nx.adjacency_matrix(data_dict[k_]["relation_graph"])

    essays, essays_labels_word, essays_labels_pos = list(), list(), list()
    essays_relation_graph, essays_node2pos, essays_adj_graph = list(), list(), list()
    essays_test, essays_test_labels_word, essays_test_labels_pos = list(), list(), list()
    essays_test_relation_graph, essays_test_node2pos, essays_test_adj_graph = list(), list(), list()
    for name_, essay_ in data_dict.items():
        if name_ in train_test_split["train"]:
            essays.append(essay_["txt"][0])
            essays_labels_word.append(essay_["entities_label_word"])
            essays_labels_pos.append(essay_["entities_label_pos"])
            essays_relation_graph.append(essay_["relation_graph"])
            essays_node2pos.append(essay_["node2pos"])
            essays_adj_graph.append(essay_["adj_graph"])
        elif name_ in train_test_split["test"]:
            essays_test.append(essay_["txt"][0])
            essays_test_labels_word.append(essay_["entities_label_word"])
            essays_test_labels_pos.append(essay_["entities_label_pos"])
            essays_test_relation_graph.append(essay_["relation_graph"])
            essays_test_node2pos.append(essay_["node2pos"])
            essays_test_adj_graph.append(essay_["adj_graph"])

    return (essays, essays_labels_word, essays_labels_pos,
            essays_relation_graph, essays_node2pos, essays_adj_graph), \
           (essays_test, essays_test_labels_word, essays_test_labels_pos,
            essays_test_relation_graph, essays_test_node2pos, essays_test_adj_graph)


def parse_txt_content(txt_content: list, lower):
    txt_content = [txt_content_.translate(trans_table) for txt_content_ in txt_content]
    if lower:
        txt_content = [txt_content_.lower() for txt_content_ in txt_content]
    title = tokenizer(txt_content[0])
    content = tokenizer(" ".join(txt_content))
    sentence = nltk.sent_tokenize(" ".join(txt_content[1:]))
    full_str = "".join(txt_content)
    return content, sentence, full_str, title


def parse_ann_content(ann_content: list, lower):
    ann_content = [ann_content_.translate(trans_table) for ann_content_ in ann_content]
    res = dict()
    for ann_content_ in ann_content:
        ann_content_ = ann_content_.replace("\n", "")
        ann_content_ = ann_content_.split("\t")
        ann_type_ = ann_content_[1].split(" ")[0]
        if ann_type_ in entities:
            article = ann_content_[2]
            if lower:
                article = article.lower()
            res[ann_content_[0]] = {"type": ann_type_, "content": ann_content_[1].split(" ")[1:], "article": article}
        elif ann_type_ in relations:
            ann_role_ = [str(ann_).partition(":")[-1] for ann_ in ann_content_[1].split(" ")[1:]]
            res[ann_content_[0]] = {"type": ann_type_, "content": ann_role_}
        elif ann_type_ in attributes:
            res[ann_content_[0]] = {"type": ann_type_, "content": ann_content_[1].split(" ")[1:]}
        else:
            raise ValueError("during loading .ann file, find unknown ann type")
    return res


# def load_data(dataset_name):
#     print('load data')
#     if dataset_name == 'stackoverflow':
#         return load_stackoverflow()
#     elif dataset_name == 'biomedical':
#         return load_biomedical()
#     elif dataset_name == 'search_snippets':
#         return load_search_snippet2()
#     else:
#         raise Exception('dataset not found...')


if __name__ == "__main__":
    load_essays()
    #
    # t = u'中国，中文，标点符号！你好？１２３４５＠＃【】+=-（）‘”'
    # t2 = t.translate(trans_table)
    # print(1)
    # s = "\"as sa d da da s d\""
    # a = nltk.word_tokenize(s)
    # for a_ in a:
    #     print(len(a_))
    print(1)