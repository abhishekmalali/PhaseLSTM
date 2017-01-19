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

class FlagsObject(object):
    def __init__(self, parse_data):
        for key in parse_data:
            setattr(self, key, parse_data[key])

n_input = 1
n_out = 3

def listdir_nohidden(path):
    list_files = []
    for f in os.listdir(path):
        if not f.startswith('.'):
            list_files.append(f)
    return list_files

def only_consider_data(obs, time):
    obs_arr = np.array(obs)
    time_arr = np.array(time)
    return np.stack([obs_arr, time_arr]).T

def create_realization(obs, obs_error, time):
    new_array = []
    obs = np.array(obs)
    for i in range(len(obs)):
        new_array.append(np.random.normal(loc=obs[i],
                                          scale=obs_error[i],
                                          size=1)[0])
    return np.stack([np.array(new_array), np.array(time)]).T

train_folder = '../data/clean-eros/train/'
valid_folder = '../data/clean-eros/valid/'
train_filenames = listdir_nohidden(train_folder)
valid_filenames = listdir_nohidden(valid_folder)
num_train = len(train_filenames)

def generate_random_batch_train(batch_size, train=True):
    g_list = []
    g_len_list = []
    r_list = []
    r_len_list = []
    Y = np.zeros((batch_size, n_out))
    use_multiple_rendition_batch = np.random.choice([True, False])
    if train is True:
        files_for_batch = np.random.choice(train_filenames, size=batch_size, replace=False)
        folder_name = train_folder
        for file_num in range(batch_size):
            with open(folder_name+files_for_batch[file_num]) as data_file:
                output = json.load(data_file)
            if use_multiple_rendition_batch is True:
                g_list.append(create_realization(output['data_g'], output['data_err_g'], output['time_g']))
                g_len_list.append(len(output['data_g']))
                r_list.append(create_realization(output['data_r'], output['data_err_r'], output['time_r']))
                r_len_list.append(len(output['data_r']))
                Y[file_num, :] = output['class_array']
            else:
                g_list.append(only_consider_data(output['data_g'], output['time_g']))
                g_len_list.append(len(output['data_g']))
                r_list.append(only_consider_data(output['data_r'], output['time_r']))
                r_len_list.append(len(output['data_r']))
                Y[file_num, :] = output['class_array']

    else:
        files_for_batch = np.random.choice(valid_filenames, size=batch_size, replace=False)
        folder_name = valid_folder
        for file_num in range(batch_size):
            with open(folder_name+files_for_batch[file_num]) as data_file:
                output = json.load(data_file)
            g_list.append(only_consider_data(output['data_g'], output['time_g']))
            g_len_list.append(len(output['data_g']))
            r_list.append(only_consider_data(output['data_r'], output['time_r']))
            r_len_list.append(len(output['data_r']))
            Y[file_num, :] = output['class_array']


    # Finding optimal lengths of matrices
    max_g_len = np.max(g_len_list)
    max_r_len = np.max(r_len_list)

    X_g = np.zeros((batch_size, max_g_len, n_input+1))
    X_r = np.zeros((batch_size, max_r_len, n_input+1))

    for i in range(batch_size):
        X_g[i, :g_len_list[i], :] = g_list[i]
        X_r[i, :r_len_list[i], :] = r_list[i]

    return X_g, X_r, Y, np.array(g_len_list), np.array(r_len_list)

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
    return relevant, initial_states

def build_model():
    #inputs
    x_g = tf.placeholder(tf.float32, [None, None, n_input + 1])
    lens_g = tf.placeholder(tf.int32, [None])
    x_r = tf.placeholder(tf.float32, [None, None, n_input + 1])
    lens_r = tf.placeholder(tf.int32, [None])
    #labels
    y = tf.placeholder(tf.float32, [None, n_out])
    # weights from input to hidden
    weights = {
        'out': tf.Variable(tf.random_normal([FLAGS.n_hidden*2, n_out], dtype=tf.float32))
    }

    biases = {
        'out': tf.Variable(tf.random_normal([n_out], dtype=tf.float32)),
    }

    print ("Compiling RNN...",)
    outputs_g, initial_states_g = RNN(x_g, lens_g, scope='g')
    outputs_r, initial_states_r = RNN(x_r, lens_r, scope='r')
    # Concatenating all the outputs for classification layer
    concat_outputs = tf.concat(1, [outputs_g, outputs_r])
    # Applying weights to ger final output
    predictions = tf.nn.bias_add(tf.matmul(concat_outputs, weights['out']), biases['out'])
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
                X_g, X_r, Y, len_g, len_r = generate_random_batch_train(FLAGS.batch_size, train=True)
                res = sess.run([optimizer, cost, accuracy, cost_summary, accuracy_summary],
                               feed_dict={x_g: X_g,
                                          x_r: X_r,
                                          y: Y,
                                          lens_g: len_g,
                                          lens_r: len_r,
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
                X_gval, X_rval, Y_val, len_gval, len_rval = generate_random_batch_train(FLAGS.batch_size, train=False)
                loss_test, acc_test, summ_cost, summ_acc = sess.run([cost,
                                            accuracy, cost_val_summary, accuracy_val_summary],
                                            feed_dict={x_g: X_gval,
                                                       x_r: X_rval,
                                                       y: Y_val,
                                                       lens_g: len_gval,
                                                       lens_r: len_rval,
                                                       })
                loss_test_ += loss_test / num_val_batch
                acc_test_ += acc_test / num_val_batch
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
            if step%10 == 0:
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
