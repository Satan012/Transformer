import tensorflow as tf
import os
import sys
import numpy as np
from tqdm import tqdm
import pickle
from nltk.tokenize import word_tokenize



"""
编码器输入：sentence + </S>
解码器输入: <S> + sentenc
解码器标签：sentence + </S>
"""

data_prefix = '../../data/gigaword/gigaword_pickle/'

train_article_path = data_prefix + 'train_x.pkl'
train_title_path = data_prefix + 'train_y.pkl'
valid_article_path = data_prefix + 'valid_x.txt'
valid_title_path = data_prefix + 'valid_y.txt'

vocab_path = '../../dictionary/gigaword/word_dict_35000.pkl'
reverse_vocab_path = '../../dictionary/gigaword/reverse_word_dict_35000.pkl'

PADDING = 0
SOURCE_PADDING_LENGTH = 50
TARGET_PADDING_LENGTH = 15


def load_vocab():
    vocabs = pickle.load(open(vocab_path, 'rb'))
    reverse_vocabs = pickle.load(open(reverse_vocab_path, 'rb'))

    return vocabs, reverse_vocabs


def load_data(kind):
    if kind == 'train':
        x = pickle.load(open(train_article_path, 'rb'))
        y = pickle.load(open(train_title_path, 'rb'))
    else:
        x = pickle.load(open(valid_article_path, 'rb'))
        y = pickle.load(open(valid_title_path, 'rb'))

    return x, y

def sentecen_decoder(target_ids, reverse_vocab):
    results = []
    for t in target_ids:
        results.append(reverse_vocab[t])
    print(' '.join(results))

if __name__ == '__main__':
    vocabs, reverse_vocabs = load_vocab()

    kind = 'train'
    data_x, data_y = load_data(kind)

    with tf.python_io.TFRecordWriter('../../data/gigaword/gigaword_tfrecord/' + kind + '_gigaword.tfrecord') as writer:
        tqbar = tqdm(total=len(data_x), desc='making {} tfrecord...'.format(kind))

        for s, t in zip(data_x, data_y):
            assert len(s) <= SOURCE_PADDING_LENGTH
            assert len(t) <= TARGET_PADDING_LENGTH
            tqbar.update(1)

            s = np.pad(s, (0, SOURCE_PADDING_LENGTH - len(s)), mode='constant', constant_values=(0, PADDING))
            # s_mask = np.pad(s_mask, (0, SOURCE_PADDING_LENGTH - len(s_mask)), mode='constant', constant_values=(0, 0))

            t_mask = np.ones_like(t)
            t_input = [2] + t[:-1]

            t_input = np.pad(t_input, (0, TARGET_PADDING_LENGTH - len(t_input)), mode='constant', constant_values=(0, PADDING))
            t = np.pad(t, (0, TARGET_PADDING_LENGTH - len(t)), mode='constant', constant_values=(0, PADDING))
            t_mask = np.pad(t_mask, (0, TARGET_PADDING_LENGTH - len(t_mask)), mode='constant', constant_values=(0, PADDING))

            example = tf.train.Example(features=tf.train.Features(feature={
                'enc_input': tf.train.Feature(int64_list=tf.train.Int64List(value=s)),
                'dec_input': tf.train.Feature(int64_list=tf.train.Int64List(value=t_input)),
                'target': tf.train.Feature(int64_list=tf.train.Int64List(value=t)),
                'target_mask': tf.train.Feature(int64_list=tf.train.Int64List(value=t_mask)),
            }))
            tf_example = example.SerializeToString()
            writer.write(tf_example)
        writer.close()
        tqbar.close()



