import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.ticker as ticker
import numpy as np
import networkx as nx
import os
import sklearn
from HParameters import *
import math
unk_str = 'unk'


def plot_attention(d, x_axis_token, y_axis_token, ac_label=None,
                   in_mask=None, word_dict=None,
                   node2pos=None, relation_graph=None, name=None):
    if not isinstance(node2pos, dict):
        temp_node2pos = dict()
        for ind, ns in enumerate(node2pos):
            if ns == (0, 0):
                break
            temp_node2pos[ind] = ns
        node2pos = temp_node2pos
    word_dict = dict(zip(word_dict.values(), word_dict.keys()))
    method_name = "mean"

    if ac_label is not None:
        if in_mask is not None:
            in_mask = np.sum(in_mask)
            if len(d.shape) == 3:
                d = d[:, :in_mask, :in_mask]
            elif len(d.shape) == 2:
                d = d[:in_mask, :in_mask]
                d = np.expand_dims(d, axis=0)
            else:
                raise ValueError("不支持该格式的attention matrix")
            x_axis_token = x_axis_token[:in_mask]
            y_axis_token = y_axis_token[:in_mask]

        atten_matrix = list()

        col = None
        index = None
        for d_ in d:
            d_, col = reduce_same_label_in_x_axis(d_, ac_label, node2pos, method_name=method_name)

            d_, index = reduce_same_label_in_x_axis(d_, ac_label, node2pos, method_name=method_name)

            d_ = d_+d_.T

            atten_matrix.append(d_)
        d = np.sum(np.array(atten_matrix), axis=0)

        show_node_str(x_axis_token, index, word_dict, node2pos)

    else:
        if word_dict is not None:
            col = [word_dict[t] if t in word_dict else unk_str for t in x_axis_token]  # 需要显示的词
            index = [word_dict[t] if t in word_dict else unk_str for t in y_axis_token]  # 需要显示的词
        else:
            col = x_axis_token
            index = y_axis_token

    show_attention_matrix_with_label(d, col, index, relation_graph, name)


def trans_pos2node(pos, node2pos) -> str:
    if node2pos is not None:
        pos2node = dict(zip(node2pos.values(), node2pos.keys()))
        if pos in pos2node:
            res = pos2node[pos]
        else:
            raise KeyError("没有这个pos对应的node信息")
    else:
        res = "{}_{}".format(pos[0], pos[1])
    return res


def show_node_str(x_axis_token, index, word_dict, node2pos):
    if word_dict is not None:
        content = [word_dict[t] if t in word_dict else unk_str for t in x_axis_token]  # 需要显示的词
        if node2pos is None:
            for ind_ in index:
                l_, _, r_ = ind_.rpartition("_")
                print("{}: {}".format(ind_, " ".join(content[int(l_): int(r_)])))
        else:
            for ind_ in index:
                l_, r_ = node2pos[ind_]
                print("{}: {}".format(ind_, " ".join(content[int(l_): int(r_)])))


def reduce_same_label_in_x_axis(attention_matrix, x_axis_label, node2pos, transpose=True, do_softmax=False,
                                method_name="mean"):
    col = list()
    all_atten = list()
    if method_name == "mean":
        method = np.mean
    elif method_name == "sum":
        method = np.sum
    else:
        raise ValueError("没有该方法名：{}".format(method_name))

    for i_, x_ in enumerate(attention_matrix):
        atten = list()
        if node2pos is None:
            prior_ind_ = 0
            for ind_ in range(len(x_) - 1):
                if x_axis_label[ind_] != x_axis_label[ind_ + 1]:
                    atten.append(np.mean(np.array(x_[prior_ind_: ind_ + 1])))
                    if i_ == 0:
                        col.append(trans_pos2node((prior_ind_, ind_), node2pos))
                    prior_ind_ = ind_ + 1
        else:
            for pos in node2pos.values():
                prior_ind_, ind_ = pos
                atten.append(method(np.array(x_[prior_ind_: ind_])))
                if i_ == 0:
                    col.append(trans_pos2node((prior_ind_, ind_), node2pos))
        all_atten.append(atten)
    attention_matrix = np.array(all_atten)
    if do_softmax:
        attention_matrix = softmax(attention_matrix, axis=1)
        pass
    if transpose:
        attention_matrix = attention_matrix.transpose()
    return attention_matrix, col


def show_attention_matrix_with_label(attention_matrix, col, index, relation_graph, name):
    df = pd.DataFrame(attention_matrix, columns=col, index=index)

    fig = plt.figure(figsize=(12, 6))
    if relation_graph is None:
        ax = fig.add_subplot(111)
    else:
        ax = fig.add_subplot(121)

    cax = ax.matshow(df, interpolation='nearest', cmap='hot_r')
    # cax = ax.matshow(df)
    fig.colorbar(cax)

    tick_spacing = 1
    ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))

    # fontdict = {'rotation': 'vertical'}    #设置文字旋转
    fontdict = {'rotation': 90}  # 或者这样设置文字旋转
    # ax.set_xticklabels([''] + list(df.columns), rotation=90)  #或者直接设置到这里
    # Axes.set_xticklabels(labels, fontdict=None, minor=False, **kwargs)
    ax.set_xticklabels([''] + list(df.columns), fontdict=fontdict)
    ax.set_yticklabels([''] + list(df.index))

    if relation_graph is not None:
        ax = fig.add_subplot(122)
        ax.grid(True)
        if isinstance(relation_graph, nx.Graph):
            relation_graph_np = nx.to_numpy_matrix(relation_graph, nodelist=np.arange(len(attention_matrix)))
        else:
            len_origin = int(math.sqrt(len(relation_graph)))
            relation_graph_np = np.reshape(np.array(relation_graph), (len_origin, len_origin))
            relation_graph_np = relation_graph_np[:len(col), :len(col)]
        df = pd.DataFrame(relation_graph_np, columns=col, index=index)
        cax = ax.matshow(df, interpolation='nearest', cmap='hot_r')
        # cax = ax.matshow(df)
        fig.colorbar(cax)
        tick_spacing = 1
        ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))

        # fontdict = {'rotation': 'vertical'}    #设置文字旋转
        fontdict = {'rotation': 90}  # 或者这样设置文字旋转
        # ax.set_xticklabels([''] + list(df.columns), rotation=90)  #或者直接设置到这里
        # Axes.set_xticklabels(labels, fontdict=None, minor=False, **kwargs)
        ax.set_xticklabels([''] + list(df.columns), fontdict=fontdict)
        ax.set_yticklabels([''] + list(df.index))
    if name is None:
        plt.show()
    else:
        plt.savefig(os.path.join(RES_PATH, name))
        plt.close()


def softmax(matrix: np.ndarray, axis=-1):
    dim = len(matrix.shape) - 1
    if axis == -1:
        axis = dim
    assert dim >= axis, "axis {} must be smaller than dim {}.".format(axis, dim)
    denominator = np.sum(np.exp(matrix), axis, keepdims=True)
    return np.exp(matrix) / denominator


# TODO(shilly): level alpha matching won't work, need fix
def level_alpha_matching(labels, preds, node2pos, alpha=1.):
    assert alpha <= 1., "alpha should not be lager than 1"
    if not isinstance(node2pos, dict):
        temp_node2pos = dict()
        for ind, ns in enumerate(node2pos):
            if ns == (0, 0):
                break
            temp_node2pos[ind] = ns
        node2pos = temp_node2pos
    score = 0
    ac_level_labels = list()
    ac_level_preds = list()
    for pos in node2pos.values():
        prior_ind_, ind_ = pos
        a_score = np.mean(np.equal(labels[prior_ind_: ind_], preds[prior_ind_: ind_]))
        if a_score >= alpha:
            score += 1
            ac_level_preds.append(1)
        else:
            ac_level_preds.append(0)
        ac_level_labels.append(1)
    return score, len(node2pos), ac_level_preds, ac_level_labels


def batch_level_alpha_matching(labels, preds, node2poss, alpha=1.):
    score, n = 0., 0.
    ac_level_preds, ac_level_labels = list(), list()
    for label, target, node2pos in zip(labels, preds, node2poss):
        temp_score, temp_n, temp_ac_level_preds, temp_ac_level_labels = level_alpha_matching(label, target, node2pos, alpha)
        score += temp_score
        n += temp_n
        ac_level_preds.extend(temp_ac_level_preds)
        ac_level_labels.extend(temp_ac_level_labels)
    return score / n, sklearn.metrics.f1_score(ac_level_labels, ac_level_preds)


def compute_f1_token_basis(predictions, correct, O_Label=0):
    prec = compute_precision_token_basis(predictions, correct, O_Label)
    rec = compute_precision_token_basis(correct, predictions, O_Label)

    f1 = 0
    if (rec + prec) > 0:
        f1 = 2.0 * prec * rec / (prec + rec)

    return prec, rec, f1


def compute_precision_token_basis(guessed_sentences, correct_sentences, O_Label):
    assert (len(guessed_sentences) == len(correct_sentences))
    correctCount = 0
    count = 0

    for sentenceIdx in range(len(guessed_sentences)):
        guessed = guessed_sentences[sentenceIdx]
        correct = correct_sentences[sentenceIdx]
        assert (len(guessed) == len(correct))
        for idx in range(len(guessed)):

            if guessed[idx] != O_Label:
                count += 1

                if guessed[idx] == correct[idx]:
                    correctCount += 1

    precision = 0
    if count > 0:
        precision = float(correctCount) / count

    return precision


if __name__ == "__main__":
    pass


