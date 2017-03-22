import tensorflow as tf
import numpy as np
import pandas as pd
import ujson as json
import os
import sklearn.metrics as metrics
from tqdm import tqdm

model_name = 'ckpts/trial_folded2/folded2-mr-100-200-20-64-4_epoch_31'
batch_size = 64
save_folder = 'evals/'
n_out = 3
n_input = 1

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

def create_eval_frame(file_name, y_net, out_net, file_list_net):
    df = pd.DataFrame()
    df['Label'] = y_net
    df['Predicted'] = out_net
    df['Name'] = file_list_net
    df['Classified'] = df['Label'] == df['Predicted']
    df.to_csv(save_folder+file_name+'.csv', index=False)

def create_meta_dict(file_name, y_net, out_net, cost_net):
    accuracy = metrics.accuracy_score(y_net, out_net)
    conf_mat = metrics.confusion_matrix(y_net, out_net)
    f1_score = metrics.f1_score(y_net, out_net, average= 'weighted')
    recall_score = metrics.recall_score(y_net, out_net, average= 'weighted')
    precision_score = metrics.precision_score(y_net, out_net, average= 'weighted')
    metrics_dict = {}
    metrics_dict['Accuracy'] = accuracy
    metrics_dict['Confusion_Matrix'] = conf_mat.tolist()
    metrics_dict['Cost'] = cost_net
    metrics_dict['F1Score'] = f1_score
    metrics_dict['Recall'] = recall_score
    metrics_dict['Precision'] = precision_score
    with open(save_folder+file_name+'.json', 'w') as fp:
        json.dump(metrics_dict, fp)

test_folder = '../data/clean-eros/test/'
test_filenames = listdir_nohidden(test_folder)

def generate_test_batch(batch_size, batch_num, last_batch=False):
    g_list = []
    g_len_list = []
    r_list = []
    r_len_list = []
    Y = np.zeros((batch_size, n_out))
    folder_name = test_folder
    if last_batch is False:
        files_for_batch = test_filenames[batch_num*batch_size:(batch_num+1)*batch_size]
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
    else:
        files_for_batch = test_filenames[batch_num*batch_size:]
        for file_num in range(len(files_for_batch)):
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
        num_files_last_batch = len(files_for_batch)
        for i in range(len(files_for_batch)):
            X_g[i, :g_len_list[i], :] = g_list[i]
            X_r[i, :r_len_list[i], :] = r_list[i]
        g_len_list = g_len_list + [0]*(batch_size - num_files_last_batch)
        r_len_list = r_len_list + [0]*(batch_size - num_files_last_batch)

    return X_g, X_r, Y, np.array(g_len_list), np.array(r_len_list), files_for_batch


def restore_score_model():
    saver = tf.train.import_meta_graph(model_name+str('.meta'),
                                       clear_devices=True)
    # We can now access the default graph where all our metadata has been loaded
    graph = tf.get_default_graph()
    num_batches = int((len(test_filenames)/batch_size))+1
    eval_save_name = model_name.split('/')[2]

    with tf.Session() as sess:
    #Restoring the model
        saver.restore(sess, model_name)
        for i in tqdm(range(num_batches)):
            if i == 0:
                X_g, X_r, Y, len_g, len_r, file_list = generate_test_batch(batch_size, i)
                out_concat, accuracy, cost = sess.run(["BiasAdd:0", "Mean_1:0", "Mean:0"],
                                                   feed_dict={"Placeholder:0": X_g,
                                                              "Placeholder_2:0": X_r,
                                                              "Placeholder_1:0": len_g,
                                                              "Placeholder_3:0": len_r,
                                                              "Placeholder_4:0": Y})
                accuracy_net = accuracy
                cost_net = cost
                y_concat = Y
                file_list_net = file_list
            if i == num_batches-1:
                X_g, X_r, Y, len_g, len_r, file_list = generate_test_batch(batch_size, i, last_batch=True)

                out, accuracy, cost = sess.run(["BiasAdd:0", "Mean_1:0", "Mean:0"],
                                            feed_dict={"Placeholder:0": X_g,
                                                       "Placeholder_2:0": X_r,
                                                       "Placeholder_1:0": len_g,
                                                       "Placeholder_3:0": len_r,
                                                       "Placeholder_4:0": Y})
                cost_net = (cost_net*i + cost)/(i+1)
                accuracy_net = (accuracy_net*i + accuracy)/(i+1)
                out_concat = np.append(out_concat, out[:len(file_list), :], axis=0)
                y_concat = np.append(y_concat, Y[:len(file_list), :], axis=0)
                file_list_net = file_list_net + file_list

            else:
                X_g, X_r, Y, len_g, len_r, file_list = generate_test_batch(batch_size, i)
                out, accuracy, cost = sess.run(["BiasAdd:0", "Mean_1:0", "Mean:0"],
                                            feed_dict={"Placeholder:0": X_g,
                                                       "Placeholder_2:0": X_r,
                                                       "Placeholder_1:0": len_g,
                                                       "Placeholder_3:0": len_r,
                                                       "Placeholder_4:0": Y})
                cost_net = (cost_net*i + cost)/(i+1)
                accuracy_net = (accuracy_net*i + accuracy)/(i+1)
                out_concat = np.append(out_concat, out, axis=0)
                y_concat = np.append(y_concat, Y, axis=0)
                file_list_net = file_list_net + file_list

        out_net = np.argmax(out_concat, axis=1)
        y_net = np.argmax(y_concat, axis=1)
        create_eval_frame(eval_save_name, y_net, out_net, file_list_net)
        create_meta_dict(eval_save_name, y_net, out_net, cost_net)

if __name__ == '__main__':
    restore_score_model()
