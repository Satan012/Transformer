import tensorflow as tf
import numpy as np
from Model_Multi_GPU import Model
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1'


def make_dict(data):
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

def cal_loss():
    loss = tf.contrib.seq2seq.sequence_loss(
        model.logits,
        model.label_y,
        tf.cast(model.y_mask, tf.float32)
    )
    tf.add_to_collection("losses", loss)

    return tf.add_n(tf.get_collection('losses'), name='total_loss')


def tower_loss(scope, batch_data):
    _ = cal_loss()

    losses = tf.get_collection('losses', scope)

    total_loss = tf.add_n(losses, name='total_loss')

    return total_loss


def average_gradient(tower_grads):
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = []
        for g, _ in grad_and_vars:
            # Add 0 dimension to the gradients to represent the tower.
            expanded_g = tf.expand_dims(g, 0)

            # Append on a 'tower' dimension which we will average over below.
            grads.append(expanded_g)

        # Average over the 'tower' dimension.
        grad = tf.concat(grads, 0)
        grad = tf.reduce_mean(grad, 0)

        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads


def train(learning_rate, decay_steps, decay_rate):
    with tf.Graph().as_default(), tf.device('/cpu:0'):
        global_step = tf.train.create_global_step()


        lr = tf.train.exponential_decay(learning_rate,
                                        global_step,
                                        decay_steps,
                                        decay_rate,
                                        staircase=True)

        opt = tf.train.AdamOptimizer(lr)

        # iter = model.train_iterator

        tower_grads = []
        with tf.variable_scope(tf.get_variable_scope()):
            for i in range(model.num_gpu):
                with tf.device('/gpu:%d' % i):
                    with tf.name_scope('transformer_tower_%d' % i) as scope:
                        # Dequeues one batch for the GPU
                        dataset = model.train_dataset

                        loss = tower_loss(scope)

                        tf.get_variable_scope().reuse_variables()

                        # Retain the summaries from the final tower.
                        summaries = tf.get_collection(tf.GraphKeys.SUMMARIES,
                                                      scope)

                        # Calculate the gradients for the batch of data on this
                        grads = opt.compute_gradients(loss, gate_gradients=0)

                        # Keep track of the gradients across all towers.
                        tower_grads.append(grads)

        # We must calculate the mean of each gradient. Note that this is the
        # synchronization point across all towers.
        grads = average_gradient(tower_grads)

        train_op = opt.apply_gradients(grads, global_step=global_step)

        # The op for initializing the variables.
        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

        sess.run(init_op)
        while True:
            dataset




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
        save_summaries_secs=None,
        global_step=model.global_step,

    )
    summary_writer = tf.summary.FileWriter(logdir=log_dir)
    with sv.prepare_or_wait_for_session() as sess:
        min_loss = 100
        count = 0
        for epoch in range(100):
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
                        if loss < min_loss:
                            min_loss = loss
                            sv.saver.save(sess, save_path=log_dir, global_step=global_step)
                        summary_writer.add_summary(merged)
                except tf.errors.OutOfRangeError:
                    break
    print("Done")
