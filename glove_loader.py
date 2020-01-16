# 用gensim打开glove词向量需要在向量的开头增加一行：所有的单词数 词向量的维度
import gensim
import os
from tqdm import tqdm
import shutil
from sys import platform

from HParameters import *


# 计算行数，就是单词数
def getFileLineNums(filename):
    with open(filename, 'r', encoding="utf-8") as f:
        count = 0
        for line in f:
            count += 1
    return count


# Linux或者Windows下打开词向量文件，在开始增加一行
def prepend_line(infile, outfile, line):
    with open(infile, 'r') as old:
        with open(outfile, 'w') as new:
            new.write(str(line) + "\n")
            shutil.copyfileobj(old, new)


def prepend_slow(infile, outfile, line):
    with open(infile, 'r', encoding="utf-8") as fin:
        with open(outfile, 'w', encoding="utf-8") as fout:
            fout.write(line + "\n")
            for line in tqdm(fin):
                fout.write(line)


def trans_to_gensim_from_glove(filename, vector_dim=300):
    filename = filename.rpartition(".txt")[0]
    gensim_file = './glove/{}_gensim.txt'.format(filename)
    filename = './glove/{}.txt'.format(filename)
    num_lines = getFileLineNums(filename)
    gensim_first_line = "{} {}".format(num_lines, vector_dim)
    # Prepends the line.
    if platform == "linux" or platform == "linux2":
        prepend_line(filename, gensim_file, gensim_first_line)
    else:
        prepend_slow(filename, gensim_file, gensim_first_line)


def load_glove(filename=GLOVE_NAME):
    filename = filename.rpartition(".txt")[0]
    gensim_file = './glove/{}_gensim.txt'.format(filename)
    model = gensim.models.KeyedVectors.load_word2vec_format(gensim_file)
    return model


if __name__ == "__main__":
    # trans_to_gensim_from_glove(GLOVE_NAME, vector_dim=VECTOR_DIM)
    # model = load_glove()
    # print(model.vocab["the"])
    # print(model["the"])
    #
    # print(model.vocab["unk"])
    a = ["asdsa"]
    gensim.models.word2vec
    print(1)
