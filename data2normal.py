import text_helper as th
import data_loader as dl
import random
import os
import math
from HParameters import *


class DataSet:
    def __init__(self, train_x=None, train_y=None, test_x=None, test_y=None):
        self.train_x = train_x
        self.train_y = train_y
        self.test_x = test_x
        self.test_y = test_y

    def gen_next_batch(self, batch_size, is_train_set, epoch=None, iteration=None):
        if is_train_set is True:
            x = self.train_x
            y = self.train_y
        else:
            x = self.test_x
            y = self.test_y

        assert len(x) >= batch_size, "batch size must be smaller than data size {}.".format(len(x))

        if epoch is not None:
            until = math.ceil(float(epoch * len(x)) / float(batch_size))
        elif iteration is not None:
            until = iteration
        else:
            assert False, "epoch or iteration must be set."

        iter_ = 0
        index_list = [i for i in range(len(x))]
        while iter_ <= until:
            idxs = random.sample(index_list, batch_size)
            iter_ += 1
            yield (x[idxs], y[idxs], idxs)


class EssayV2(DataSet):
    def __init__(self):
        super().__init__()
        train_data, test_data = dl.load_essays(lower=True)
        train_texts, train_labels, train_labels_pos, train_relation_graph, train_node2pos, train_adj_graph = train_data
        test_texts, test_labels, test_labels_pos, test_relation_graph, test_node2pos, test_adj_graph = test_data
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

        if GLOVE:
            word_dict_ = dict()
            for k_, v_ in word_dict.vocab.items():
                word_dict_[k_] = v_.index
            word_dict = word_dict_

        import pickle
        with open(os.path.join(ROOT_PATH, WORD_DICT_NAME), 'wb') as f:
            pickle.dump(word_dict, f)

        self.train_x = train_tokens
        self.train_y = train_labels, train_labels_pos, train_adj_graph
        self.test_x = test_tokens
        self.test_y = test_labels, test_labels_pos, test_adj_graph

        self.train_relation_graph = train_relation_graph
        self.train_node2pos = train_node2pos
        self.test_node2pos = test_node2pos
        self.test_relation_graph = test_relation_graph

        self.num_classes_entities = len(dl.entities) - 1
        self.num_classes_pos = len(dl.pos) - 1
        self.num_classes_relations = len(dl.relations) - 1


if __name__ == "__main__":
    # train_data, test_data = dl.load_essays(lower=True)
    # train_texts, train_labels, train_labels_pos, train_relation_graph, train_node2pos, train_adj_graph = train_data
    # test_texts, test_labels, test_labels_pos, test_relation_graph, test_node2pos, test_adj_graph = test_data
    # if GLOVE:
    #     logger("loading glove")
    #     word_dict = th.load_glove()
    #     import numpy as np
    #     np.save(os.path.join(ROOT_PATH, EMBEDDING_NAME), word_dict.vectors)
    # else:
    #     word_dict = th.build_dictionary(train_texts)
    # train_tokens = th.text_to_numbers(train_texts, word_dict, glove=GLOVE)
    # test_tokens = th.text_to_numbers(test_texts, word_dict, glove=GLOVE)
    #
    # del train_texts, test_texts, train_data, test_data
    #
    # train_tokens, train_labels, train_labels_pos, train_relation_graph, train_node2pos, train_adj_graph
    # test_tokens, test_labels, test_labels_pos, test_relation_graph, test_node2pos, test_adj_graph
    #
    # if GLOVE:
    #     word_dict_ = dict()
    #     for k_, v_ in word_dict.vocab.items():
    #         word_dict_[k_] = v_.index
    #     word_dict = word_dict_
    #
    # import pickle
    # with open(os.path.join(ROOT_PATH, WORD_DICT_NAME), 'wb') as f:
    #     pickle.dump(word_dict, f)
    pass