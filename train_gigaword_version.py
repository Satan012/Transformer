import tensorflow as tf
import numpy as np
from Model import Model
import os
import json

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('train_set_path', 'data/gigaword/gigaword_tfrecord/train_gigaword.tfrecord', 'the path of cnn_train set')
tf.app.flags.DEFINE_string('valid_set_path', 'data/gigaword/gigaword_tfrecord/valid_gigaword.tfrecord', 'the path of validation set')
tf.app.flags.DEFINE_string('test_set_path', 'data/gigaword/gigaword_tfrecord/valid_gigaword.tfrecord', 'the path of test set')
tf.app.flags.DEFINE_integer('epoch', 100, 'epoch of training step')
tf.app.flags.DEFINE_integer('batch_size', 512, 'mini_batch size')
tf.app.flags.DEFINE_integer('input_max_len', 50, 'max text length of input')
tf.app.flags.DEFINE_integer('target_max_len', 15, 'max text length of target')
tf.app.flags.DEFINE_integer('vocab_size', 35000, 'size of vocab')
tf.app.flags.DEFINE_integer('embedding_size', 256, 'embedding size of word')
tf.app.flags.DEFINE_string('model_state', 'cnn_train', 'model state')
tf.app.flags.DEFINE_boolean('debugging', True, 'debugging or not')
tf.app.flags.DEFINE_integer('head_num', 8, 'number of head')
tf.app.flags.DEFINE_float('lr', 0.0001, 'learning rate')
tf.app.flags.DEFINE_integer('num_blocks', 6, 'number of block')
tf.app.flags.DEFINE_boolean('sinusoid', True, 'whether use to positional embedding')
tf.app.flags.DEFINE_string('logdir', 'log/giga_train_2/', 'dir to save model')

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def make_dict(model, data):
    batch_size = np.shape(data['target'])[0]
    d = {
        model.input_x: data['enc_input'],
        model.input_y: data['dec_input'],
        model.y_mask: data['target_mask'],
        model.label_y: data['target'],
        model.dropout_keep_prob: 0.1,
        model.batch_size: batch_size
    }
    return d


if __name__ == '__main__':
    # print('Building Dictionary...')
    # word_dict, reversed_dict, article_max_len, summary_max_len = giga_utils.build_dict()
    # print('Building DataSet...')
    # train_x, train_y = giga_utils.build_dataset('cnn_train', word_dict, article_max_len, summary_max_len)
    # valid_x, valid_y = giga_utils.build_dataset('valid', word_dict, article_max_len, summary_max_len)
    # exit(0)
    model = Model(FLAGS)
    # Construct graph
    print("Graph loaded...")

    log_dir = model.hps.logdir

    sv = tf.train.Supervisor(
        logdir=log_dir,
        save_model_secs=0,
        save_summaries_secs=None,
        global_step=model.global_step
    )

    exp_id = model.hps.logdir.split('_')[-1][:-1]
    txtLogName = 'exp_{}.txt'.format(exp_id)

    # if os.path.exists('log/txtLog/gigaword/' + txtLogName):
    #     print('file has exist !!!')
    #     exit(0)

    f = open('log/txtLog/gigaword/' + txtLogName, 'a')
    d = dict(FLAGS.flag_values_dict())
    json.dump(d, f, indent=4)
    f.flush()
    f.write('\n\n-----------------------------------\n\n')

    summary_writer = tf.summary.FileWriter(logdir=log_dir)
    with sv.prepare_or_wait_for_session() as sess:
        min_loss = 100
        count = 0
        for epoch in range(model.hps.epoch):
            print('--------- epoch: {} ---------'.format(epoch))
            sess.run(model.train_iterator.initializer)
            while True:
                try:
                    train_data = sess.run(model.train_dataset)
                    feed_dict = make_dict(model, train_data)
                    _, loss, global_step, merged = sess.run(
                        [model.train_op, model.mean_loss, model.global_step, model.merged], feed_dict)
                    # print('{} --> {} --> {}'.format(epoch, global_step, loss))
                    if global_step % 50 == 0:
                        print('{} --> {} --> {}'.format(epoch, global_step, loss))
                        f.write('{} --> {} --> {}\n'.format(epoch, global_step, loss))
                        f.flush()
                        if loss < min_loss:
                            min_loss = loss
                            sv.saver.save(sess, save_path=log_dir, global_step=global_step)
                        summary_writer.add_summary(merged)
                except tf.errors.OutOfRangeError:
                    break
    print("Done")
