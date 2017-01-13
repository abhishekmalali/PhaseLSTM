import tensorflow as tf
import numpy as np
from tabulate import tabulate
from tensorflow.python.ops.rnn_cell import LSTMCell, GRUCell
from PhasedLSTMCell import PhasedLSTMCell, multiPLSTM
from data_generation import create_batch_dataset
from tqdm import tqdm
import pandas as pd
flags = tf.flags
flags.DEFINE_string("unit", "PLSTM", "Can be PSLTM, LSTM, GRU")
flags.DEFINE_integer("n_hidden", 20, "hidden units in the recurrent layer")
flags.DEFINE_integer("n_epochs", 500, "number of epochs")
flags.DEFINE_integer("batch_size", 32, "batch size")
flags.DEFINE_integer("b_per_epoch", 10, "batches per epoch")
flags.DEFINE_integer("n_layers", 2, "hidden units in the recurrent layer")
flags.DEFINE_float("exp_init", 3., "Value for initialization of Tau")
flags.DEFINE_string('train_ckpt', 'ckpts/trial/model_ini.ckpt', 'Train checkpoint file')
flags.DEFINE_string('train_logs', 'tmp/trial/', 'Log directory')
FLAGS = flags.FLAGS

n_input = 1
n_out = 2

def RNN(_X, _weights, _biases, lens):
    if FLAGS.unit == "PLSTM":
        cell = PhasedLSTMCell(FLAGS.n_hidden, use_peepholes=True, state_is_tuple=True)
    elif FLAGS.unit == "GRU":
        cell = GRUCell(FLAGS.n_hidden)
    elif FLAGS.unit == "LSTM":
        cell = LSTMCell(FLAGS.n_hidden, use_peepholes=True, state_is_tuple=True)
    else:
        raise ValueError("Unit '{}' not implemented.".format(FLAGS.unit))
    initial_states = [tf.nn.rnn_cell.LSTMStateTuple(tf.zeros([FLAGS.batch_size, FLAGS.n_hidden], tf.float32), tf.zeros([FLAGS.batch_size, FLAGS.n_hidden], tf.float32)) for _ in range(FLAGS.n_layers)]
    outputs, initial_states = multiPLSTM(_X, FLAGS.batch_size, lens, FLAGS.n_layers, FLAGS.n_hidden, n_input, initial_states)

    outputs = tf.slice(outputs, [0, 0, 0], [-1, -1, FLAGS.n_hidden])

    batch_size = tf.shape(outputs)[0]
    max_len = tf.shape(outputs)[1]
    out_size = int(outputs.get_shape()[2])
    index = tf.range(0, batch_size) * max_len + (lens - 1)
    flat = tf.reshape(outputs, [-1, out_size])
    relevant = tf.gather(flat, index)

    return tf.nn.bias_add(tf.matmul(relevant, _weights['out']), _biases['out']), initial_states

def build_model():
    x = tf.placeholder(tf.float32, [None, None, n_input + 1])
    lens = tf.placeholder(tf.int32, [None])
    #labels
    y = tf.placeholder(tf.float32, [None, n_out])
    # weights from input to hidden
    weights = {
        'out': tf.Variable(tf.random_normal([FLAGS.n_hidden, n_out], dtype=tf.float32))
    }

    biases = {
        'out': tf.Variable(tf.random_normal([n_out], dtype=tf.float32))
    }

    # Register weights to be monitored by tensorboard
    w_out_hist = tf.summary.histogram("weights_out", weights['out'])
    b_out_hist = tf.summary.histogram("biases_out", biases['out'])
    print ("Compiling RNN...",)
    predictions, initial_states = RNN(x, weights, biases, lens)
    print ("DONE!")
    # Register initial_states to be monitored by tensorboard
    initial_states_hist = tf.summary.histogram("initial_states", initial_states[0][0])
    print ("Compiling cost functions...",)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(predictions, y))
    print ("DONE!")
    cost_summary = tf.summary.scalar("cost", cost)
    cost_val_summary = tf.summary.scalar("cost_val", cost)
    optimizer = tf.train.AdamOptimizer().minimize(cost)
    # evaluation
    correct_pred = tf.equal(tf.argmax(predictions, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    accuracy_summary = tf.summary.scalar("accuracy", accuracy)
    accuracy_val_summary = tf.summary.scalar("accuracy_val", accuracy)

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    columns = ['Epoch', 'train_cost', 'train_acc', 'val_cost', 'val_acc']
    log_df = pd.DataFrame(data=np.zeros((0,len(columns))), columns=columns)
    with tf.Session(config=config) as sess:
        sess.run(init)
        writer = tf.summary.FileWriter(FLAGS.train_logs, sess.graph)
        for step in range(FLAGS.n_epochs):
            train_cost = 0
            train_acc = 0
	    batch_xs, batch_ys, leng = create_batch_dataset(FLAGS.batch_size)
            for i in tqdm(range(FLAGS.b_per_epoch)):
            	res = sess.run([optimizer, cost, accuracy, cost_summary, accuracy_summary],
                               feed_dict={x: batch_xs,
                                          y: batch_ys,
                                          lens: leng
                                          })
                writer.add_summary(res[3], step * FLAGS.b_per_epoch + i)
                writer.add_summary(res[4], step * FLAGS.b_per_epoch + i)
                train_cost += res[1] / FLAGS.b_per_epoch
                train_acc += res[2] / FLAGS.b_per_epoch
            print "Epoch "+ str(step+1) +" train_cost: "+str(train_cost)+" train_accuracy: "+str(train_acc)
            batch_val_xs, batch_val_ys, leng_val = create_batch_dataset(FLAGS.batch_size)
            loss_test, acc_test, summ_cost, summ_acc = sess.run([cost,
                                            accuracy, cost_val_summary, accuracy_val_summary],
                                            feed_dict={x: batch_val_xs,
                                                       y: batch_val_ys,
                                                       lens: leng_val})
            writer.add_summary(summ_cost, step * FLAGS.b_per_epoch + i)
            writer.add_summary(summ_acc, step * FLAGS.b_per_epoch + i)
            table = [["Train", train_cost, train_acc],
                     ["Test", loss_test, acc_test]]
            headers = ["Epoch={}".format(step), "Cost", "Accuracy"]
            log_df = log_df.append({'Epoch': step+1,
                                    'train_cost': train_cost,
                                    'train_acc': train_acc,
                                    'val_cost': loss_test,
                                    'val_acc': acc_test},
                                    ignore_index = True)
            print (tabulate(table, headers, tablefmt='grid'))
            log_df.to_csv('log_trial.csv')
        saver.save(sess, FLAGS.train_ckpt)

def main(argv=None):
    with tf.device('/gpu:0'):
        build_model()

if __name__ == '__main__':
  tf.app.run()
