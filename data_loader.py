# -*- coding: utf-8 -*-
import os
from collections import Counter

import nltk
import numpy as np
import scipy.io
import string
from zhon.hanzi import punctuation
import unicodedata
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.preprocessing import MinMaxScaler

log_or_print = print
entities = {"MajorClaim": 1, "Claim": 2, "Premise": 3}
relations = {"supports": 1, "attacks": 2}
attributes = {"Stance": 1}
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


def load_essays(data_path="./data/ArgumentAnnotatedEssays-2.0"):
    detail_data_path = os.path.join(data_path, "brat-project-final")
    data_list = os.listdir(detail_data_path)
    accept_types = ["txt", "ann", ]
    data_dict = dict()
    log_or_print("loading data path...")
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
                log_or_print("during loading data path: unknown type: {}".format(data_))
    log_or_print("loading data content...")
    for k_, v_ in data_dict.items():
        for accept_type_ in accept_types:
            data_name_ = v_[accept_type_]
            with open(os.path.join(detail_data_path, data_name_), 'r', encoding="utf-8") as f_:
                data_content_ = f_.readlines()
                if accept_type_ == "txt":
                    data_dict[k_][accept_type_] = parse_txt_content(data_content_)
                elif accept_type_ == "ann":
                    data_dict[k_][accept_type_] = parse_ann_content(data_content_)
                else:
                    log_or_print("during loading data content: unknown type: {}".format(accept_type_))
    for k_, v_ in data_dict.items():
        data_dict[k_]["entities_label_char"] = np.zeros(len(v_["txt"][2]), dtype=np.int)
        data_dict[k_]["entities_label_word"] = np.zeros(len(v_["txt"][0]), dtype=np.int)
        for a_k_, a_v_ in v_["ann"].items():
            if a_v_["type"] in entities:
                end = int(a_v_["content"][1])
                start = int(a_v_["content"][0])
                debug_ = v_["txt"][2][start:end]
                assert debug_ == a_v_["article"]
                data_dict[k_]["entities_label_char"][start: end] = entities[a_v_["type"]]
        now_ind_ = 0
        for ind_, word_ in enumerate(v_["txt"][0]):

            if word_[0] in string.punctuation or word_[0] in punctuation:
                now_ind_ -= 1
            if word_ == "n\'t":
                now_ind_ -= 1

            while v_["txt"][2][now_ind_] in string.whitespace:
                now_ind_ += 1

            label_ = data_dict[k_]["entities_label_char"][now_ind_: now_ind_ + len(word_)]
            debug_ = v_["txt"][2][now_ind_: now_ind_ + len(word_)]
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

            now_ind_ += (len(word_)+1)

    essays = [(eassy_["txt"][0], eassy_["entities_label_word"]) for eassy_ in data_dict.values()]
    return essays


def parse_txt_content(txt_content: list):
    txt_content = [txt_content_.translate(trans_table) for txt_content_ in txt_content]
    title = nltk.word_tokenize(txt_content[0])
    content = nltk.word_tokenize(" ".join(txt_content))
    sentence = nltk.sent_tokenize(" ".join(txt_content[1:]))
    full_str = "".join(txt_content)
    return content, sentence, full_str, title


def parse_ann_content(ann_content: list):
    ann_content = [ann_content_.translate(trans_table) for ann_content_ in ann_content]
    res = dict()
    for ann_content_ in ann_content:
        ann_content_ = ann_content_.replace("\n", "")
        ann_content_ = ann_content_.split("\t")
        ann_type_ = ann_content_[1].split(" ")[0]
        if ann_type_ in entities:
            res[ann_content_[0]] = {"type": ann_type_, "content": ann_content_[1].split(" ")[1:], "article": ann_content_[2]}
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
    # print(1)