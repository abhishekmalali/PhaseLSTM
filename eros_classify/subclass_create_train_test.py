import pandas as pd
import ujson as json
from sklearn.model_selection import train_test_split
import numpy as np
import os
from tqdm import tqdm
import subprocess

files_to_process = ['RRL_catalog_crossmatch_eros_ogle.csv',
                    'EB_catalog_crossmatch_eros_ogle.csv',
                    'CEPH_catalog_crossmatch_eros_ogle.csv']
csv_path = '../data/EROS/'
save_path = '../data/clean-eros/subclasses/'
class_list = []
for file_name in files_to_process:
    df = pd.read_csv(csv_path+file_name)
    add_class_list = pd.unique(df['subclass'])[pd.value_counts(df['subclass']) > 100]
    class_list = class_list + list(add_class_list)


numclasses = len(class_list)
class_dict = dict(zip(class_list, range(numclasses)))
#Saving the class details dictionary
with open(save_path + 'class_details_dictionary.json', 'w') as fp:
    json.dump(class_dict_dict, fp)

def create_augmented_signal(data, err, time, period, num_periods=5):
    new_time = np.mod(time, period)
    len_signal = len(data)
    for i in range(num_periods):
        choice = np.random.randint(low=0, high=len_signal, size=int(len_signal/2))
        if i == 0:
            out_signal = data[choice]
            out_err = err[choice] = err[choice]
            out_time = new_time[choice]
        else:
            out_signal = np.append(data[choice], out_signal)
            out_err = np.append(err[choice], out_err)
            out_time = np.append((i*period)+new_time[choice], out_time)
    #Sorting the arrays
    idx_ = np.argsort(out_time)
    out_time = out_time[idx_]
    out_signal = out_signal[idx_]
    out_err = out_err[idx_]
    return out_time.tolist(), out_signal.tolist(), out_err.tolist()

def parse_files(file_name, period, class_type, type_='train'):
    try:
        command = 'find ../data/EROS/OGLE_trainingDB | grep -i '+file_name+'.time'
        file_loc = subprocess.check_output(command, shell=True).split('\n')[0]
    except:
        return None

    #Finding the class value
    try:
        class_value = class_dict[class_type]
    except:
        return None

    with open(file_loc) as f:
        time_array_g = []
        time_array_r = []
        arr_g = []
        arr_r = []
        arr_err_g = []
        arr_err_r = []
        for line in f:
            if line.find('#') == -1:
                out = line.strip().split(' ')
                out = filter(None, out)
                out = map(float, out)
                if out[1] != 99.999:
                    arr_g.append(out[1])
                    arr_err_g.append(out[2])
                    time_array_g.append(out[0])
                if out[3] != 99.999:
                    arr_r.append(out[3])
                    arr_err_r.append(out[4])
                    time_array_r.append(out[0])
        # Normalizing the time scale and starting from 0
        min_time = min(min(time_array_g), min(time_array_r))
        time_array_g[:] = [time - min_time for time in time_array_g]
        time_array_r[:] = [time - min_time for time in time_array_r]
        # standardizing the data
        arr_g = np.array(arr_g)
        mean_arr_g = np.mean(arr_g)
        std_arr_g = np.std(arr_g)
        arr_r = np.array(arr_r)
        mean_arr_r = np.mean(arr_r)
        std_arr_r = np.std(arr_r)

        data_dict = {}
        data_g = ((arr_g - mean_arr_g)/std_arr_g)
        data_r = ((arr_r - mean_arr_r)/std_arr_r)
        data_err_g = (arr_err_g/std_arr_g)
        data_err_r = (arr_err_r/std_arr_r)
        data_dict['time_g'], data_dict['data_g'], data_dict['data_err_g'] = \
                       create_augmented_signal(data_g, data_err_g, time_array_g, period, num_periods=5)
        data_dict['time_r'], data_dict['data_r'], data_dict['data_err_r'] = \
                       create_augmented_signal(data_r, data_err_r, time_array_r, period, num_periods=5)
        class_array = [0]*numclasses
        class_array[class_value] = 1
        data_dict['class_array'] = class_array
        data_dict['class_value'] = class_value
        #Appending the period for additional data
        data_dict['period'] = period
    path = '../data/clean-eros/subclasses/'+type_+'/'
    with open(path + file_name + '.json', 'w') as fp:
        json.dump(data_dict, fp)


def create_train_test():
    folders = ['train', 'test', 'valid']
    for folder in folders:
        folderpath = save_path + folder
        if not os.path.exists(folderpath):
            os.makedirs(folderpath)
    for key in files_to_process:
        print "Computing for :" + key
        df = pd.read_csv(csv_path+key)
        train_df, test_df = train_test_split(df, train_size=0.75, random_state=50)
        train_df2, valid_df = train_test_split(train_df, train_size=0.7, random_state=70)

        train_df2[['EROS_id','period','subclass']].apply(lambda x : parse_files(*x,
                                                        type_='train'), axis=1)
        valid_df[['EROS_id','period','subclass']].apply(lambda x : parse_files(*x,
                                                        type_='valid'), axis=1)
        test_df[['EROS_id','period','subclass']].apply(lambda x : parse_files(*x,
                                                        type_='test'), axis=1)

if __name__ == "__main__":
    create_train_test()
