from collections import namedtuple
import shutil
from modules import *
import os




class Model(object):
    def setHps(self, FLAGS):
        d = dict(FLAGS.flag_values_dict())
        hps = namedtuple('HParams', list(d.keys()))
        hps = hps._make(list(d.values()))
        return hps

    def parse_function(self, example_proto):
        parsed_example = tf.parse_single_example(
            example_proto,
            features={
                'enc_input': tf.VarLenFeature(dtype=tf.int64),
                'dec_input': tf.VarLenFeature(dtype=tf.int64),
                'target': tf.VarLenFeature(dtype=tf.int64),
                'target_mask': tf.VarLenFeature(dtype=tf.int64)
            }
        )
        result = {
            'enc_input': tf.sparse_tensor_to_dense(tf.cast(parsed_example['enc_input'], tf.int32)),
            'dec_input': tf.sparse_tensor_to_dense(tf.cast(parsed_example['dec_input'], tf.int32)),
            'target': tf.sparse_tensor_to_dense(tf.cast(parsed_example['target'], tf.int32)),  # 标签
            'target_mask': tf.sparse_tensor_to_dense(tf.cast(parsed_example['target_mask'], tf.int32))
        }
        return result

    def getDataset(self, data_path, mode='train'):
        dataset = tf.data.TFRecordDataset(data_path)
        dataset = dataset.map(self.parse_function)
        if mode == 'train':
            dataset = dataset.shuffle(1000000).batch(self.hps.batch_size)
        else:
            dataset = dataset.batch(self.hps.batch_size)
        iterator = dataset.make_initializable_iterator()
        return iterator

    def __init__(self, FLAGS, is_training=True):
        self.hps = self.setHps(FLAGS)
        self.global_step = tf.train.create_global_step()
        self.train_iterator = self.getDataset(self.hps.train_set_path)
        self.train_dataset = self.train_iterator.get_next()
        self.valid_iterator = self.getDataset(self.hps.valid_set_path, 'valid')
        self.valid_dataset = self.valid_iterator.get_next()
        self.test_iterator = self.getDataset(self.hps.test_set_path, 'test')
        self.test_dataset = self.test_iterator.get_next()
        if self.hps.debugging:
            if os.path.exists(self.hps.logdir):
                print('remove：' + self.hps.logdir)
                shutil.rmtree(self.hps.logdir)

        # Placeholders for input, output and dropout
        self.input_x = tf.placeholder(tf.int32, [None, self.hps.input_max_len], name="input_x")
        self.input_y = tf.placeholder(tf.int32, [None, self.hps.target_max_len], name='input_y')
        self.label_y = tf.placeholder(tf.int32, [None, self.hps.target_max_len], name="label_y")
        self.y_mask = tf.placeholder(tf.int32, [None, self.hps.target_max_len], name='y_mask')

        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        self.batch_size = tf.placeholder(tf.int32, name='batch_size')

        # self.lr = tf.train.exponential_decay(self.hps.lr,
        #                             self.global_step,
        #                             10000,
        #                             0.8,
        #                             staircase=True)
        self.lr = self.hps.lr

        # Encoder
        with tf.variable_scope("encoder"):
            # Embedding
            self.enc = embedding(self.input_x,
                                 vocab_size=self.hps.vocab_size,
                                 num_units=self.hps.embedding_size,
                                 scale=True,
                                 scope="enc_embed")
            print('shape-->{}: {}'.format('self.enc', np.shape(self.enc)))  # [batch_size, enc_max_len, embedding_size]

            # Positional Encoding, 将positional embedding和 word embedding直接相加
            self.enc_pos = self.enc + tf.cast(positional_encoding(self.input_x,
                                                        self.batch_size,
                                                        num_units=self.hps.embedding_size,
                                                        zero_pad=False,
                                                        scale=False,
                                                        scope="enc_pe"), dtype=tf.float32)

            # Dropout
            self.enc_pos = tf.layers.dropout(self.enc_pos,
                                         rate=self.dropout_keep_prob,
                                         training=tf.convert_to_tensor(is_training))

            # Blocks
            for i in range(self.hps.num_blocks):
                with tf.variable_scope("num_blocks_{}".format(i)):
                    # Multihead Attention
                    self.enc = multihead_attention(queries=self.enc_pos,
                                                   keys=self.enc_pos,
                                                   unpos_key=self.enc,
                                                   unpos_query=self.enc,
                                                   num_units=self.hps.embedding_size,
                                                   num_heads=self.hps.head_num,
                                                   dropout_rate=self.dropout_keep_prob,
                                                   is_training=is_training,
                                                   causality=False)
                    # Feed Forward
                    self.enc_pos = feedforward(self.enc_pos,
                                           num_units=[4 * self.hps.embedding_size, self.hps.embedding_size])  # [batch, seq, hidden]
            print('shape-->{}: {}'.format('multi_self.enc', np.shape(self.enc)))
        # Decoder
        with tf.variable_scope("decoder"):
            # Embedding
            self.dec = embedding(self.input_y,
                                 vocab_size=self.hps.vocab_size,
                                 num_units=self.hps.embedding_size,
                                 scale=True,
                                 scope="dec_embed")

            ## Positional Encoding
            self.dec_pos = self.dec + tf.cast(positional_encoding(self.input_y,
                                                                  self.batch_size,
                                                                  num_units=self.hps.embedding_size,
                                                                  zero_pad=False,
                                                                  scale=False,
                                                                  scope="dec_pe"), dtype=tf.float32)


            # Dropout
            self.dec_pos = tf.layers.dropout(self.dec_pos,
                                         rate=self.dropout_keep_prob,
                                         training=tf.convert_to_tensor(is_training))

            # Blocks
            for i in range(self.hps.num_blocks):
                with tf.variable_scope("num_blocks_{}".format(i)):
                    ## Multihead Attention ( self-attention)
                    self.dec_pos = multihead_attention(queries=self.dec_pos,
                                                   keys=self.dec_pos,
                                                   unpos_key=self.dec,
                                                   unpos_query=self.dec,
                                                   num_units=self.hps.embedding_size,
                                                   num_heads=self.hps.head_num,
                                                   dropout_rate=self.dropout_keep_prob,
                                                   is_training=is_training,
                                                   causality=True,
                                                   scope="self_attention")  # [batch_size, target_len, embedding]

                    ## Multihead Attention ( vanilla attention)
                    self.dec_pos = multihead_attention(queries=self.dec_pos,
                                                   keys=self.enc_pos,
                                                   unpos_key=self.enc,
                                                   unpos_query=self.dec,
                                                   num_units=self.hps.embedding_size,
                                                   num_heads=self.hps.head_num,
                                                   dropout_rate=self.dropout_keep_prob,
                                                   is_training=is_training,
                                                   causality=False,
                                                   scope="vanilla_attention")  # [batch_size, target_len, embedding]

                    # Feed Forward, [batch_size, target_len, embedding]
                    self.dec_pos = feedforward(self.dec_pos, num_units=[4 * self.hps.embedding_size, self.hps.embedding_size])

        # Final linear projection
        self.logits = tf.layers.dense(self.dec_pos, self.hps.vocab_size)  # [batch_size, target_len, vocab_size]
        self.preds = tf.to_int32(tf.arg_max(self.logits, dimension=-1))

        if is_training:
            # Loss
            self.loss = tf.contrib.seq2seq.sequence_loss(
                    self.logits,
                    self.label_y,
                    tf.cast(self.y_mask, tf.float32)
                )
            # self.mean_loss = tf.reduce_sum(self.loss * tf.cast(self.y_mask, dtype=tf.float32))
            self.mean_loss = self.loss
            # Training Scheme
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
            self.train_op = self.optimizer.minimize(self.mean_loss, global_step=self.global_step)

            # Summary
            tf.summary.scalar('mean_loss', self.mean_loss)
            self.merged = tf.summary.merge_all()


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    model = Model()
