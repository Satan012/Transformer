#coding:utf-8
import os

from nltk.tokenize import word_tokenize
import re
import collections
import pickle
import numpy as np
from gensim.models.keyedvectors import KeyedVectors
from gensim.test.utils import get_tmpfile
from gensim.scripts.glove2word2vec import glove2word2vec
from tqdm import tqdm


dataset_prefix = '/media/satan/3e75fcb2-d0e7-4b40-9145-34692ff66ff1/sunmingsheng/DataSet/GigaWord/'
dictionary_prefix = 'dictionary/gigaword/'
gigaword_pickle_prefix = 'data/giga/gigaword_pickle/'

train_article_path = dataset_prefix + "train.article.txt"
train_title_path = dataset_prefix + "train.title.txt"
valid_article_path = dataset_prefix + "valid.article.filter.txt"
valid_title_path = dataset_prefix + "valid.title.filter.txt"

dict_length = 35000  # 预定字典长度


def clean_str(sentence):
    sentence = re.sub("[#.!?\'\"`]+", "#", sentence)
    return sentence


def get_text_list(data_path):
    with open(data_path, "r") as f:
        return list(map(lambda x: clean_str(x.strip()), f.readlines()))


def build_dict():
    filePath = dictionary_prefix + 'word_dict_' + str(dict_length) + '.pkl'
    if os.path.isfile(filePath):
        print('\n{} has exited...'.format(filePath))
        with open(filePath, "rb") as f:
            word_dict = pickle.load(f)
            print('len_word_dict:', len(word_dict))
            f.close()
    else:
        train_article_list = get_text_list(train_article_path)
        train_title_list = get_text_list(train_title_path)
        print('finished get_text_list')
        words = list()
        total_len = len(train_article_list + train_title_list)
        tqbar = tqdm(total=total_len, desc='word_tokenizing')
        for idx, sentence in enumerate(train_article_list + train_title_list):
            for word in word_tokenize(sentence):
                words.append(word)
            tqbar.update(1)
        tqbar.close()

        word_counter = collections.Counter(words).most_common()
        word_dict = collections.OrderedDict()
        word_dict["<padding>"] = 0
        word_dict["<unk>"] = 1
        word_dict["<s>"] = 2
        word_dict["</s>"] = 3
        for word, _ in word_counter:
            if len(word_dict) < dict_length:
                word_dict[word] = len(word_dict)
            else:
                break
        assert len(word_dict) == dict_length
        with open(filePath, "wb") as f:
            pickle.dump(word_dict, f)

    reversed_dict = collections.OrderedDict(zip(word_dict.values(), word_dict.keys()))

    article_max_len = 50
    summary_max_len = 15

    return word_dict, reversed_dict, article_max_len, summary_max_len


def build_dataset(step, word_dict, article_max_len, summary_max_len):
    """
    :param step: train or valid, will return different dataset
    :param word_dict:
    :param article_max_len: 50
    :param summary_max_len: 15
    :return:
    """
    prefix = gigaword_pickle_prefix + step + '_'
    if os.path.isfile(prefix + 'x.pkl') and os.path.isfile(prefix + 'y.pkl'):
        print('already have %s dataset...' % step)
        x = pickle.load(open(prefix + 'x.pkl', 'rb'))
        y = pickle.load(open(prefix + 'y.pkl', 'rb'))
    else:
        if step == "train":
            print('enter train dataset preparing...')
            article_list = get_text_list(train_article_path)
            title_list = get_text_list(train_title_path)
        elif step == "valid":
            print('enter valid dataset preparing...')
            article_list = get_text_list(valid_article_path)
            title_list = get_text_list(valid_title_path)
        else:
            raise NotImplementedError

        x = list(map(lambda d: word_tokenize(d), article_list))  # 对原文进行分词处理
        x = list(map(lambda d: list(map(lambda w: word_dict.get(w, word_dict["<unk>"]), d)), x))  # 对原文进行映射，word-->id
        x = list(map(lambda d: np.concatenate([d[:article_max_len-1], [word_dict["</s>"]]]), x))
        # x = list(map(lambda d: d + (article_max_len - len(d)) * [word_dict["<padding>"]], x))  # 对于原文进行padding

        y = list(map(lambda d: word_tokenize(d), title_list))  # 分词
        y = list(map(lambda d: list(map(lambda w: word_dict.get(w, word_dict["<unk>"]), d)), y))
        y = list(map(lambda d: d[:(summary_max_len - 1)], y))  # label没有被pad

        pickle.dump(x, open(prefix + 'x.pkl', 'wb'))
        pickle.dump(y, open(prefix + 'y.pkl', 'wb'))

    return x, y

"""
def ducDataset(step, word_dict, article_max_len, summary_max_len, toy=False):
    prefix = step + '_cnn_'
    if os.path.isfile(prefix + 'x.pkl') and os.path.isfile(prefix + 'y.pkl'):
        print('already have %s dataset...' % step)
        x = pickle.load(open(prefix + 'x.pkl', 'rb'))
        y = pickle.load(open(prefix + 'y.pkl', 'rb'))

    else:
        if step == 'duc2003':
            print('enter duc2003')
            article_list = get_text_list(duc2003_path, toy)
            title_list = get_text_list(duc2003_path_title, toy)
            # print(article_list)
            # print(title_list)
        elif step == 'duc2004':
            print('enter duc2004')
            article_list = get_text_list(duc2004_path, toy)
            title_list = get_text_list(duc2004_path_title, toy)
        else:
            raise NotImplementedError

        x = list(map(lambda d: word_tokenize(d), article_list))
        x = list(map(lambda d: list(map(lambda w: word_dict.get(w, word_dict["<unk>"]), d)), x))
        x = list(map(lambda d: d[:article_max_len], x))
        x = list(map(lambda d: d + (article_max_len - len(d)) * [word_dict["<padding>"]], x))

        y = list(map(lambda d: word_tokenize(d), title_list))  # 分词
        y = list(map(lambda d: list(map(lambda w: word_dict.get(w, word_dict["<unk>"]), d)), y))
        y = list(map(lambda d: d[:(summary_max_len - 1)], y))  # label没有被pad

        pickle.dump(x, open(prefix + 'x.pkl', 'wb'))
        pickle.dump(y, open(prefix + 'y.pkl', 'wb'))
    return x, y
"""


def batch_iter(inputs, outputs, batch_size, num_epochs):
    tmp_input = []
    tmp_output = []
    for i, o in zip(inputs, outputs):
        if i[0] != 0 and o[0] != 0:
            tmp_input.append(i)
            tmp_output.append(o)
    inputs, outputs = tmp_input, tmp_output
    inputs = np.array(inputs)
    outputs = np.array(outputs)

    num_batches_per_epoch = (len(inputs) - 1) // batch_size + 1
    for epoch in range(num_epochs):
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, len(inputs))
            yield inputs[start_index:end_index], outputs[start_index:end_index], epoch


def get_init_embedding(reversed_dict, embedding_size):
    glove_file = "/home/sd1/mkl/code/rcnn_mkl/data/glove.6B/glove.6B.300d.txt"
    word2vec_file = get_tmpfile("word2vec_format.vec")
    glove2word2vec(glove_file, word2vec_file)
    print("Loading Glove vectors...")
    word_vectors = KeyedVectors.load_word2vec_format(word2vec_file)

    print('finished load')

    count = 0
    word_vec_list = list()
    for _, word in sorted(reversed_dict.items()):
        try:
            word_vec = word_vectors.word_vec(word)
        except KeyError:
            count += 1
            word_vec = np.zeros([embedding_size], dtype=np.float32)

        word_vec_list.append(word_vec)

    # Assign random vector to <s>, </s> token
    word_vec_list[2] = np.random.normal(0, 1, embedding_size)
    word_vec_list[3] = np.random.normal(0, 1, embedding_size)

    print('total count:', count)
    np.save('50000embedding.npy', np.array(word_vec_list))

    return np.array(word_vec_list)


def sortLength(fileName):
    with open(fileName, 'r') as f:
        lines = f.readlines()
        lengths = []
        for l in lines:
            l = l.strip().split()
            lengths.append(len(l))
        lengths = sorted(lengths, reverse=True)
        f.close()
    print(np.mean(lengths))
