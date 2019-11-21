# Text Helper Functions
# ---------------------------------------
#
# We pull out text helper functions to reduce redundant code

import string
import os
import collections
from tqdm import tqdm


# Normalize text
def normalize_text(text):
    # Lower case
    text = text.lower()
    # Remove punctuation
    text = ''.join(c if c not in string.punctuation else ' '+c for c in text)
    # Trim extra whitespace
    text = ' '.join(text.split())
    return text


# Build dictionary of words
def build_dictionary(sentences, vocabulary_size=60000):
    # Turn sentences (list of strings) into lists of words
    split_sentences = sentences
    words = [x for sublist in split_sentences for x in sublist]
    # Initialize list of [word, word_count] for each word, starting with unknown
    count = [['[UNK]', -1]]
    # Now add most frequent words, limited to the N-most frequent (N=vocabulary size)
    count.extend(collections.Counter(words).most_common())
    # Now create the dictionary
    word_dict = {'[PAD]': 0, '[BEGIN]': 1, '[EOS]': 2, '[CLS]': 3, '[SEP]': 4, '[MASK]': 5}
    # For each word, that we want in the dictionary, add it, then make it
    # the value of the prior dictionary length
    for word, word_count in tqdm(count):
        word_dict[word] = len(word_dict)
        if len(word_dict) >= vocabulary_size:
            break
    return word_dict


# Turn text data into lists of integers from dictionary
def text_to_numbers(sentences, word_dict, ensure=None):
    # Initialize the returned data
    data = []
    for i, sentence in tqdm(enumerate(sentences)):
        sentence_data = []
        # For each word, either use selected index or rare word index
        split_sentences = sentence

        for word in split_sentences:
            if word in word_dict:
                word_ix = word_dict[word]
            else:
                word_ix = word_dict['[UNK]']

            sentence_data.append(word_ix)

        if ensure is None:
            data.append(sentence_data)
        else:
            if ensure[i] == len(sentence_data):
                data.append(sentence_data)
            else:
                print(ensure[i], " doesn't match ", sentence_data)
    return data


# Turn text data into lists of integers from dictionary
def numbers_to_text(sentences, word_dict):
    word_dict = dict(zip(word_dict.values(), word_dict.keys()))
    # Initialize the returned data
    data = []
    for sentence in sentences:
        sentence_data = []
        # For each word, either use selected index or rare word index
        split_sentences = sentence
        for word in split_sentences:
            if word in word_dict:
                word_ix = word_dict[word]
            else:
                word_ix = '[UNK]'
            sentence_data.append(word_ix)
        data.append(sentence_data)
    return data

