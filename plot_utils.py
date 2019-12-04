import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.ticker as ticker
import numpy as np
import networkx as nx

unk_str = 'unk'


def plot_attention(d, x_axis_token, y_axis_token, ac_label=None,
                   in_mask=None, word_dict=None,
                   node2pos=None, relation_graph=None):
    word_dict = dict(zip(word_dict.values(), word_dict.keys()))

    if ac_label is not None:
        if in_mask is not None:
            in_mask = np.sum(in_mask)
            d = d[:in_mask, :in_mask]
            x_axis_token = x_axis_token[:in_mask]
            y_axis_token = y_axis_token[:in_mask]

        d, col = sum_same_label_in_x_axis(d, ac_label, node2pos)

        d, index = sum_same_label_in_x_axis(d, ac_label, node2pos)

        show_node_str(x_axis_token, index, word_dict, node2pos)

    else:
        if word_dict is not None:
            col = [word_dict[t] if t in word_dict else unk_str for t in x_axis_token]  # 需要显示的词
            index = [word_dict[t] if t in word_dict else unk_str for t in y_axis_token]  # 需要显示的词
        else:
            col = x_axis_token
            index = y_axis_token

    show_attention_matrix_with_label(d, col, index, relation_graph)


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


def sum_same_label_in_x_axis(attention_matrix, x_axis_label, node2pos, transpose=True):
    col = list()
    all_atten = list()
    for i_, x_ in enumerate(attention_matrix):
        atten = list()
        prior_ind_ = 0
        for ind_ in range(len(x_) - 1):
            if x_axis_label[ind_] != x_axis_label[ind_ + 1]:
                atten.append(np.mean(np.array(x_[prior_ind_: ind_ + 1])))
                if i_ == 0:
                    col.append(trans_pos2node((prior_ind_, ind_ + 1), node2pos))
                prior_ind_ = ind_ + 1
        all_atten.append(atten)
    attention_matrix = np.array(all_atten)
    if transpose:
        attention_matrix = attention_matrix.transpose()
    return attention_matrix, col


def show_attention_matrix_with_label(attention_matrix, col, index, relation_graph):
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
        relation_graph_np = nx.to_numpy_matrix(relation_graph, nodelist=np.arange(len(attention_matrix)))
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

    plt.show()


