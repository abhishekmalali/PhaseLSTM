import tensorflow as tf
import numpy as np
from tabulate import tabulate
from tensorflow.python.ops.rnn_cell import LSTMCell, GRUCell
from PhasedLSTMCell import PhasedLSTMCell, multiPLSTM
from tensorflow.python.ops import variable_scope as vs
import ujson as json
from tqdm import tqdm
import pandas as pd
import os
import argparse

"""
flags = tf.flags
flags.DEFINE_string("unit", "PLSTM", "Can be PSLTM, LSTM, GRU")
flags.DEFINE_integer("n_hidden", 100, "hidden units in the recurrent layer")
flags.DEFINE_integer("n_epochs", 100, "number of epochs")
flags.DEFINE_integer("batch_size", 32, "batch size")
flags.DEFINE_integer("b_per_epoch", 200, "batches per epoch")
flags.DEFINE_integer("n_layers", 5, "hidden units in the recurrent layer")
flags.DEFINE_float("exp_init", 3., "Value for initialization of Tau")
flags.DEFINE_string('train_ckpt', 'ckpts/trial/model_ini.ckpt', 'Train checkpoint file')
flags.DEFINE_string('train_logs', 'tmp/trial/', 'Log directory')
FLAGS = flags.FLAGS
"""

class FlagsObject(object):
    def __init__(self, parse_data):
        for key in parse_data:
            setattr(self, key, parse_data[key])

n_input = 1
n_out = 2

def listdir_nohidden(path):
    list_files = []
    for f in os.listdir(path):
        if not f.startswith('.'):
            list_files.append(f)
    return list_files

train_folder = '../data/snpcc/train/'
valid_folder = '../data/snpcc/valid/'
train_filenames = listdir_nohidden(train_folder)
valid_filenames = listdir_nohidden(valid_folder)
num_train = len(train_filenames)

def generate_random_batch_train(batch_size, train=True):
    if train is True:
        files_for_batch = np.random.choice(train_filenames, size=batch_size, replace=False)
        folder_name = train_folder
    else:
        files_for_batch = np.random.choice(valid_filenames, size=batch_size, replace=False)
        folder_name = valid_folder
    g_list = []
    g_len_list = []
    r_list = []
    r_len_list = []
    i_list = []
    i_len_list = []
    z_list = []
    z_len_list = []
    Y = np.zeros((batch_size, n_out))
    for file_num in range(batch_size):
        with open(folder_name+files_for_batch[file_num]) as data_file:
            output = json.load(data_file)
        g_list.append(output['obs_g'])
        g_len_list.append(len(output['obs_g']))
        r_list.append(output['obs_r'])
        r_len_list.append(len(output['obs_r']))
        i_list.append(output['obs_i'])
        i_len_list.append(len(output['obs_i']))
        z_list.append(output['obs_z'])
        z_len_list.append(len(output['obs_z']))
        Y[file_num, :] = output['labels']

    # Finding optimal lengths of matrices
    max_g_len = np.max(g_len_list)
    max_r_len = np.max(r_len_list)
    max_i_len = np.max(i_len_list)
    max_z_len = np.max(z_len_list)

    X_g = np.zeros((batch_size, max_g_len, n_input+1))
    X_r = np.zeros((batch_size, max_r_len, n_input+1))
    X_i = np.zeros((batch_size, max_i_len, n_input+1))
    X_z = np.zeros((batch_size, max_z_len, n_input+1))

    for i in range(batch_size):
        X_g[i, :g_len_list[i], :] = g_list[i]
        X_r[i, :r_len_list[i], :] = r_list[i]
        X_i[i, :i_len_list[i], :] = i_list[i]
        X_z[i, :z_len_list[i], :] = z_list[i]

    return X_g, X_r, X_i, X_z, Y,\
        np.array(g_len_list), np.array(r_len_list),\
        np.array(i_len_list), np.array(z_len_list)

def RNN(_X, lens, scope='Network'):
    if FLAGS.unit == "PLSTM":
        cell = PhasedLSTMCell(FLAGS.n_hidden, use_peepholes=True, state_is_tuple=True)
    elif FLAGS.unit == "GRU":
        cell = GRUCell(FLAGS.n_hidden)
    elif FLAGS.unit == "LSTM":
        cell = LSTMCell(FLAGS.n_hidden, use_peepholes=True, state_is_tuple=True)
    else:
        raise ValueError("Unit '{}' not implemented.".format(FLAGS.unit))
    with vs.variable_scope(scope):
        initial_states = [tf.nn.rnn_cell.LSTMStateTuple(tf.zeros([FLAGS.batch_size, FLAGS.n_hidden], tf.float32), tf.zeros([FLAGS.batch_size, FLAGS.n_hidden], tf.float32)) for _ in range(FLAGS.n_layers)]
        outputs, initial_states = multiPLSTM(_X, FLAGS.batch_size, lens, FLAGS.n_layers, FLAGS.n_hidden, n_input, initial_states)
        outputs = tf.slice(outputs, [0, 0, 0], [-1, -1, FLAGS.n_hidden])
        batch_size = tf.shape(outputs)[0]
        max_len = tf.shape(outputs)[1]
        out_size = int(outputs.get_shape()[2])
        index = tf.range(0, batch_size) * max_len + (lens - 1)
        flat = tf.reshape(outputs, [-1, out_size])
        relevant = tf.gather(flat, index)
        """
        new_output = tf.reduce_mean(outputs, axis=1)
        print new_output.get_shape()
        """
    return relevant, initial_states

def build_model():
    #inputs
    x_g = tf.placeholder(tf.float32, [None, None, n_input + 1])
    lens_g = tf.placeholder(tf.int32, [None])
    x_r = tf.placeholder(tf.float32, [None, None, n_input + 1])
    lens_r = tf.placeholder(tf.int32, [None])
    x_i = tf.placeholder(tf.float32, [None, None, n_input + 1])
    lens_i = tf.placeholder(tf.int32, [None])
    x_z = tf.placeholder(tf.float32, [None, None, n_input + 1])
    lens_z = tf.placeholder(tf.int32, [None])
    #labels
    y = tf.placeholder(tf.float32, [None, n_out])
    # weights from input to hidden
    weights = {
        'out': tf.Variable(tf.random_normal([FLAGS.n_hidden*4, 200], dtype=tf.float32)),
        'out2': tf.Variable(tf.random_normal([200, 50], dtype=tf.float32)),
        'out3': tf.Variable(tf.random_normal([50, n_out], dtype=tf.float32))
    }

    biases = {
        'out': tf.Variable(tf.random_normal([200], dtype=tf.float32)),
        'out2': tf.Variable(tf.random_normal([50], dtype=tf.float32)),
        'out3': tf.Variable(tf.random_normal([n_out], dtype=tf.float32))
    }

    print ("Compiling RNN...",)
    outputs_g, initial_states_g = RNN(x_g, lens_g, scope='g')
    outputs_r, initial_states_r = RNN(x_r, lens_r, scope='r')
    outputs_i, initial_states_i = RNN(x_i, lens_i, scope='i')
    outputs_z, initial_states_z = RNN(x_z, lens_z, scope='z')
    # Concatenating all the outputs for classification layer
    concat_outputs = tf.concat(1, [outputs_g, outputs_r, outputs_i, outputs_z])
    # Applying weights to ger final output
    predictions_layer0 = tf.nn.bias_add(tf.matmul(concat_outputs, weights['out']), biases['out'])
    predictions_layer1 = tf.nn.sigmoid(tf.nn.bias_add(tf.matmul(predictions_layer0, weights['out2']), biases['out2']))
    predictions = tf.nn.bias_add(tf.matmul(predictions_layer1, weights['out3']), biases['out3'])
    print ("DONE!")
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
            for i in tqdm(range(FLAGS.b_per_epoch)):
                X_g, X_r, X_i, X_z, Y, len_g, len_r, len_i, len_z = generate_random_batch_train(FLAGS.batch_size, train=True)
                res = sess.run([optimizer, cost, accuracy, cost_summary, accuracy_summary],
                               feed_dict={x_g: X_g,
                                          x_r: X_r,
                                          x_i: X_i,
                                          x_z: X_z,
                                          y: Y,
                                          lens_g: len_g,
                                          lens_r: len_r,
                                          lens_i: len_i,
                                          lens_z: len_z,
                                          })
                writer.add_summary(res[3], step * FLAGS.b_per_epoch + i)
                writer.add_summary(res[4], step * FLAGS.b_per_epoch + i)
                train_cost += res[1] / FLAGS.b_per_epoch
                train_acc += res[2] / FLAGS.b_per_epoch
            print "Epoch "+ str(step+1) +" train_cost: "+str(train_cost)+" train_accuracy: "+str(train_acc)
            loss_test_ = 0
            acc_test_ = 0
            num_val_batch = 50
            for k in range(num_val_batch):
                X_gval, X_rval, X_ival, X_zval, Y_val, len_gval, len_rval, len_ival, len_zval = generate_random_batch_train(FLAGS.batch_size, train=False)
                loss_test, acc_test, summ_cost, summ_acc = sess.run([cost,
                                            accuracy, cost_val_summary, accuracy_val_summary],
                                            feed_dict={x_g: X_gval,
                                                       x_r: X_rval,
                                                       x_i: X_ival,
                                                       x_z: X_zval,
                                                       y: Y_val,
                                                       lens_g: len_gval,
                                                       lens_r: len_rval,
                                                       lens_i: len_ival,
                                                       lens_z: len_zval,
                                                       })
                loss_test_ += loss_test / num_val_batch
                acc_test_ += acc_test / num_val_batch
            # writer.add_summary(loss_test_, step)
            # writer.add_summary(acc_test_, step)
            table = [["Train", train_cost, train_acc],
                     ["Test", loss_test_, acc_test_]]
            headers = ["Epoch={}".format(step), "Cost", "Accuracy"]
            log_df = log_df.append({'Epoch': step+1,
                                    'train_cost': train_cost,
                                    'train_acc': train_acc,
                                    'val_cost': loss_test_,
                                    'val_acc': acc_test_},
                                    ignore_index = True)
            print (tabulate(table, headers, tablefmt='grid'))
            log_df.to_csv(FLAGS.quick_log)
        saver.save(sess, FLAGS.train_ckpt)


def main(argv=None):
    parser = argparse.ArgumentParser(description=' RNN using PLSTM.')

    # File and path naming stuff
    parser.add_argument('--run_id',   default=os.environ.get('LSB_JOBID',''), help='ID of the run, used in saving.')
    # Append .ini to the file
    parser.add_argument('--filename', default='model_ini', help='Filename to save model and log to.')
    parser.add_argument('--filepath', default='ckpts/trial/', help='Filepath to save model and log from.')
    # Controlling the network parameters
    parser.add_argument('--unit', default="PLSTM", help='Choose from PLSTM, LSTM or GRU.')
    parser.add_argument('--n_hidden', default=100, help='Select memory size for RNN')
    parser.add_argument('--n_epochs', default=100, help='Select number of training epochs')
    parser.add_argument('--batch_size',  default=32, help='Select batchs size')
    parser.add_argument('--b_per_epoch',  default=200, help='Select batches per epoch')
    parser.add_argument('--n_layers',  default=4, help='Select number of layers for individual LSTM network')
    parser.add_argument('--exp_init',  default=3., help='Initializer value for kronos gate')
    parser.add_argument('--quick_log_directory', default='logs/', help = 'Quick Log Directory')
    parser.add_argument('--quick_log_file', default='log_run.csv', help = 'Quick Log File')
    parser.add_argument('--log_directory', default='tmp/trial/1', help = 'Log Directory')
    args = parser.parse_args()
    # Creating the quick log file name
    log_file_name = args.quick_log_directory + args.quick_log_file
    # Creating the model save path
    model_save_path = args.filepath + args.filename + '.ckpt'

    # Checking if the file paths exist and if not creating them
    if not os.path.exists(args.quick_log_directory):
        os.makedirs(args.quick_log_directory)
    if not os.path.exists(args.filepath):
        os.makedirs(args.filepath)

    flags = {}
    flags["unit"] = "PLSTM"
    flags["n_hidden"] = int(args.n_hidden)
    flags["n_epochs"] = int(args.n_epochs)
    flags["batch_size"] = int(args.batch_size)
    flags["b_per_epoch"] = int(args.b_per_epoch)
    flags["n_layers"] = int(args.n_layers)
    flags["exp_init"] = float(args.exp_init)
    flags['train_ckpt'] = model_save_path
    flags['train_logs'] = args.log_directory
    flags['quick_log'] = log_file_name
    global FLAGS
    FLAGS = FlagsObject(flags)
    with tf.device('/gpu:0'):
        build_model()

if __name__ == '__main__':
    tf.app.run()
