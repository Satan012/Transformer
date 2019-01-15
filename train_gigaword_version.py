import tensorflow as tf
import numpy as np
from Model import Model
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '1'


def make_dict(model, data):
    batch_size = np.shape(data['target'])[0]
    d = {
        model.input_x: data['enc_input'],
        model.input_y: data['dec_input'],
        model.y_mask: data['target_mask'],
        model.label_y: data['target'],
        model.dropout_keep_prob: 0.5,
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
    model = Model()
    # Construct graph
    print("Graph loaded...")

    log_dir = 'log/giga_train/'

    sv = tf.train.Supervisor(
        logdir=log_dir,
        save_model_secs=0,
        save_summaries_secs=2,
        global_step=model.global_step
    )
    with sv.prepare_or_wait_for_session() as sess:
        min_loss = 100
        count = 0
        for epoch in range(1, 5000):
            print('--------- epoch: {} ---------'.format(epoch))
            sess.run(model.train_iterator.initializer)
            while True:
                try:
                    train_data = sess.run(model.train_dataset)
                    feed_dict = make_dict(model, train_data)
                    _, loss, global_step = sess.run(
                        [model.train_op, model.mean_loss, model.global_step], feed_dict)
                    # print('{} --> {} --> {}'.format(epoch, global_step, loss))
                    if global_step % 50 == 0:
                        print('{} --> {} --> {}'.format(epoch, global_step, loss))
                        if loss < min_loss:
                            min_loss = loss
                            sv.saver.save(sess, save_path=log_dir, global_step=global_step)
                except tf.errors.OutOfRangeError:
                    break
    print("Done")
