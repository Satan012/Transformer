import tensorflow as tf
import os
from tensor2tensor import problems
import numpy as np
from tqdm import tqdm


"""
编码器输入：sentence + </S>
解码器输入: <S> + sentenc
解码器标签：sentence + </S>
"""

def encode(input_str, max_length):
    """Input str to features dict, ready for inference"""
    inputs = encoders["inputs"].encode(input_str)

    if len(inputs) >= max_length:
        inputs = inputs[:max_length - 1]

    inputs = inputs + [1]  # add SOS(2), EOS(1) id
    return inputs


def decode(integers):
    """List of ints to str"""
    integers = list(np.squeeze(integers))
    if 1 in integers:
        integers = integers[:integers.index(1)]  # 1 is EOS id
    return encoders["inputs"].decode(np.squeeze(integers))

data_dir = '/media/satan/3e75fcb2-d0e7-4b40-9145-34692ff66ff1/sunmingsheng/DataSet/t2t/data'
tmp_dir = '/media/satan/3e75fcb2-d0e7-4b40-9145-34692ff66ff1/sunmingsheng/DataSet/t2t/tmp'
# train_dir = '/media/satan/3e75fcb2-d0e7-4b40-9145-34692ff66ff1/sunmingsheng/DataSet/t2t/google'

dataLen = {'train': 287113, 'test': 11490, 'dev': 13368}
cnn_dailymail = problems.problem('summarize_cnn_dailymail32k')
PADDING = 0
SOURCE_PADDING_LENGTH = 1000
TARGET_PADDING_LENGTH = 100

encoders = cnn_dailymail.feature_encoders(data_dir)

kind = ['test', 'train', 'dev']
for k in kind:
    source_name = 'cnndm.{}.source'.format(k)
    target_name = 'cnndm.{}.target'.format(k)
    with open(os.path.join(tmp_dir, target_name), encoding='utf-8') as f0:
        targets = f0.readlines()
        targets = [t.split('\n')[0] for t in targets]  # 去除换行符号
        for i in tqdm(range(dataLen[k]), desc=target_name):
            targets[i] = encode(targets[i], max_length=TARGET_PADDING_LENGTH)
        f0.close()

    with open(os.path.join(tmp_dir, source_name), encoding='utf-8') as f1:
        sources = f1.readlines()
        sources = [s.split('\n')[0] for s in sources]
        for i in tqdm(range(dataLen[k]), desc=source_name):
            sources[i] = encode(sources[i], max_length=SOURCE_PADDING_LENGTH)
        f1.close()

    with tf.python_io.TFRecordWriter('./cnn_tfrecord/google/' + k + '.tfrecord') as writer:
        tqbar = tqdm(total=dataLen[k], desc='making {} tfrecord...'.format(k))
        for s, t in zip(sources, targets):
            assert len(s) <= SOURCE_PADDING_LENGTH
            assert len(t) <= TARGET_PADDING_LENGTH
            tqbar.update(1)
            # s_mask = np.ones_like(s)
            s = np.pad(s, (0, SOURCE_PADDING_LENGTH - len(s)), mode='constant', constant_values=(0, PADDING))
            # s_mask = np.pad(s_mask, (0, SOURCE_PADDING_LENGTH - len(s_mask)), mode='constant', constant_values=(0, 0))

            t_mask = np.ones_like(t)
            t_input = [2] + t[:-1]

            t_input = np.pad(t_input, (0, TARGET_PADDING_LENGTH - len(t_input)), mode='constant', constant_values=(0, PADDING))
            t = np.pad(t, (0, TARGET_PADDING_LENGTH - len(t)), mode='constant', constant_values=(0, PADDING))
            t_mask = np.pad(t_mask, (0, TARGET_PADDING_LENGTH - len(t_mask)), mode='constant', constant_values=(0, 0))

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
