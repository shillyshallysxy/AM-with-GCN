import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.ticker as ticker
import numpy as np


def plot_attention(d, x_axis_token, y_axis_token, ac_label=None, in_mask=None, word_dict=None):
    word_dict = dict(zip(word_dict.values(), word_dict.keys()))
    unk_str = 'unk'

    if ac_label is not None:
        if in_mask is not None:
            in_mask = np.sum(in_mask)
            d = d[:in_mask, :in_mask]
            x_axis_token = x_axis_token[:in_mask]
            y_axis_token = y_axis_token[:in_mask]

        col = list()
        index = list()
        all_atten = list()
        for i_, x_ in enumerate(d):
            atten = list()
            prior_ind_ = 0
            for ind_ in range(len(x_)-1):
                if ac_label[ind_] != ac_label[ind_+1]:
                    atten.append(np.mean(np.array(x_[prior_ind_: ind_+1])))
                    if i_ == 0:
                        col.append("{}_{}".format(prior_ind_, ind_+1))
                    prior_ind_ = ind_+1
            all_atten.append(atten)
        d = np.array(all_atten).transpose()

        all_atten = list()
        for i_, x_ in enumerate(d):
            atten = list()
            prior_ind_ = 0
            for ind_ in range(len(x_) - 1):
                if ac_label[ind_] != ac_label[ind_ + 1]:
                    atten.append(np.mean(np.array(x_[prior_ind_: ind_ + 1])))
                    if i_ == 0:
                        index.append("{}_{}".format(prior_ind_, ind_ + 1))
                    prior_ind_ = ind_ + 1
            all_atten.append(atten)

        d = np.array(all_atten).transpose()
        if word_dict is not None:
            content = [word_dict[t] if t in word_dict else unk_str for t in x_axis_token]  # 需要显示的词
            for ind_ in index:
                l_, _, r_ = ind_.rpartition("_")
                print("{}: {}".format(ind_, " ".join(content[int(l_): int(r_)])))
    else:
        if word_dict is not None:
            col = [word_dict[t] if t in word_dict else unk_str for t in x_axis_token]  # 需要显示的词
            index = [word_dict[t] if t in word_dict else unk_str for t in y_axis_token]  # 需要显示的词
        else:
            col = x_axis_token
            index = y_axis_token

    df = pd.DataFrame(d, columns=col, index=index)

    fig = plt.figure()

    ax = fig.add_subplot(111)

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
