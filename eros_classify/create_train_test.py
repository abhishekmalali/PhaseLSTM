import pandas as pd
import ujson as json
from sklearn.model_selection import train_test_split
import numpy as np
import os
from tqdm import tqdm
import subprocess

meta_dict = {'RRL_catalog_crossmatch_eros_ogle.csv' : 0,
             'EB_catalog_crossmatch_eros_ogle.csv': 1,
             'CEPH_catalog_crossmatch_eros_ogle.csv': 2}

def parse_files(file_name, period, class_value=0, type_='train'):
    try:
        command = 'find ../data/EROS/OGLE_trainingDB | grep -i '+file_name+'.time'
        file_loc = subprocess.check_output(command, shell=True).split('\n')[0]
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
        data_dict['data_g'] = ((arr_g - mean_arr_g)/std_arr_g).tolist()
        data_dict['data_r'] = ((arr_r - mean_arr_r)/std_arr_r).tolist()
        data_dict['data_err_g'] = (arr_err_g/std_arr_g).tolist()
        data_dict['data_err_r'] = (arr_err_r/std_arr_r).tolist()
        data_dict['time_g'] = time_array_g
        data_dict['time_r'] = time_array_r
        class_array = [0, 0 ,0]
        class_array[class_value] = 1
        data_dict['class_array'] = class_array
        data_dict['class_value'] = class_value
        #Appending the period for additional data
        data_dict['period'] = period
    path = '../data/clean-eros/'+type_+'/'
    with open(path + file_name + '.json', 'w') as fp:
        json.dump(data_dict, fp)


def create_train_test():
    folders = ['train', 'test', 'valid']
    for folder in folders:
        folderpath = '../data/clean-eros/' + folder
        if not os.path.exists(folderpath):
            os.makedirs(folderpath)
    for key in meta_dict.keys():
        print "Computing for :" + key
        meta_path = '../data/EROS/'
        csv_path = meta_path + key
        df = pd.read_csv(csv_path)
        cls_val = meta_dict[key]
        train_df, test_df = train_test_split(df, train_size=0.75)
        train_df2, valid_df = train_test_split(train_df, train_size=0.7)

        train_df2[['EROS_id','period']].apply(lambda x : parse_files(*x, class_value=cls_val,
                                                        type_='train'), axis=1)
        valid_df[['EROS_id','period']].apply(lambda x : parse_files(*x, class_value=cls_val,
                                                        type_='valid'), axis=1)
        test_df[['EROS_id','period']].apply(lambda x : parse_files(*x, class_value=cls_val,
                                                        type_='test'), axis=1)

if __name__ == "__main__":
    create_train_test()
