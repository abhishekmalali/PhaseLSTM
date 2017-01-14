import os
import numpy as np
import pandas as pd
from itertools import groupby
from sklearn.model_selection import train_test_split
import ujson as json

#File paths
datapath = '../data/snpcc/data/'
train_path = '../data/snpcc/train/'
test_path = '../data/snpcc/test/'
valid_path = '../data/snpcc/valid/'
flux_norm = 1.
time_norm = 1.

def listdir_nohidden(path):
    list_files = []
    for f in os.listdir(path):
        if not f.startswith('.'):
            list_files.append(f)
    return list_files

# Not including the error measurements here.
def parse_files(filename, datapath):

    with open(datapath+filename) as f:
        #Initializing output to NoneType
        survey = snid = sntype = ra = decl = hostGalId = hostz = \
        redSpec = simType = simRedShift = None
        first_obs = None
        # Data format [magnitude, time]
        obs_g = []
        obs_r = []
        obs_i = []
        obs_z = []
        for line in f:
            s = line.split(':')
            if len(s) > 0:
                if s[0] == 'SURVEY':
                    survey = s[1].strip()
                if s[0] == 'SNID':
                    snid = int(s[1].strip())
                if s[0] == 'SNTYPE':
                    sntype = int(s[1].strip())
                if s[0] == 'RA':
                    ra = float(s[1].split('deg')[0].strip())
                if s[0] == 'DECL':
                    decl = float(s[1].split('deg')[0].strip())
                if s[0] == 'HOST_GALAXY_GALID':
                    hostGalId = int(s[1].strip())
                if s[0] == 'HOST_GALAXY_PHOTO-Z':
                    hostz = float(s[1].split('+-')[0].strip()),\
                            float(s[1].split('+-')[1].strip())
                if s[0] == 'REDSHIFT_SPEC':
                    redSpec = float(s[1].split('+-')[0].strip()),\
                            float(s[1].split('+-')[1].strip())
                if s[0] == 'SIM_COMMENT':
                    simType =  s[1].split(',')[0].split('=')[1].strip()
                if s[0] == 'SIM_REDSHIFT':
                    simRedShift = float(s[1].strip())
                if s[0] == 'OBS':
                    o = s[1].split()
                    if first_obs is None:
                        first_obs = float(o[0])
                    time = (float(o[0]) - first_obs)/time_norm
                    if o[1] == 'g':
                        g = float(o[3])/flux_norm
                        #g_error = float(o[4])/flux_norm
                        obs_g.append([g, time])
                    elif o[1] == 'r':
                        r = float(o[3])/flux_norm
                        #r_error = float(o[4])/flux_norm
                        obs_r.append([r, time])
                    elif o[1] == 'i':
                        i = float(o[3])/flux_norm
                        #i_error = float(o[4])/flux_norm
                        obs_i.append([i, time])
                    elif o[1] == 'z':
                        z = float(o[3])/flux_norm
                        #z_error = float(o[4])/flux_norm
                        obs_z.append([z, time])

        return {"survey":survey, "snid":snid, "sntype":sntype,\
                "ra": ra, "decl":decl, "hostGalId": hostGalId,\
                "hostz": hostz, "redSpec": redSpec, "simType": simType,\
                "simRedShift": simRedShift, "obs_g": obs_g, "obs_r": obs_r,\
                "obs_i": obs_i, "obs_z": obs_z}

def save_prep_data(file_data):
    out = [0, 0]
    if file_data["simType"] == 'Ia':
        class_ = 0
        out[0] = 1
    else:
        class_ = 1
        out[1] = 1
    data_dictionary = {}
    data_dictionary['labels'] = out
    data_dictionary['class_'] = class_
    data_dictionary['obs_g'] = file_data['obs_g']
    data_dictionary['obs_r'] = file_data['obs_r']
    data_dictionary['obs_i'] = file_data['obs_i']
    data_dictionary['obs_z'] = file_data['obs_z']

    return data_dictionary

def parse_files_for_splitting(filename, datapath):

    with open(datapath+filename) as f:
        #Initializing output to NoneType
        survey = simtype = None
        for line in f:
            s = line.split(':')
            if len(s) > 0:
                if s[0] == 'SURVEY':
                    survey = s[1].strip()
                if s[0] == 'SIM_COMMENT':
                    simType =  s[1].split(',')[0].split('=')[1].strip()
                    if simType == 'Ia':
                        sntype = 0
                    else:
                        sntype = 1

        return {'filename': filename, 'sntype': sntype, 'survey': survey}

def save_from_dataframe(filename, path=train_path):
    mega_meta_dict = parse_files(filename, datapath)
    data_dictionary = save_prep_data(mega_meta_dict)
    #Saving the file as a json with the same name
    #Splitting the name of the file inorder to remove filetype
    file_prefix = filename.split('.')[0]
    with open(path + file_prefix + '.json', 'w') as fp:
        json.dump(data_dictionary, fp)

if __name__ == '__main__':
    fileList = listdir_nohidden(datapath)
    dataframe = pd.DataFrame(columns=['survey', 'sntype'])
    for fileName in fileList:
        features = parse_files_for_splitting(fileName, datapath)
        dataframe = dataframe.append(pd.Series(features), ignore_index=True)
    #Splitting the dataset
    train_df, test_df = train_test_split(dataframe, test_size=0.75)
    train_df2, valid_df = train_test_split(train_df, train_size=0.7)
    # Printing test and train dataset state
    print "Number of train curves :"+ str(len(train_df2))
    print "Train data distributions"
    print pd.value_counts(train_df2['sntype'])
    print "==================================="
    print "Number of validation curves :"+ str(len(valid_df))
    print "Validation data distributions"
    print pd.value_counts(valid_df['sntype'])
    print "==================================="
    print "Number of test curves :"+ str(len(test_df))
    print "Test data distributions"
    print pd.value_counts(test_df['sntype'])


    train_df2['filename'].apply(lambda x: save_from_dataframe(x, path=train_path))
    valid_df['filename'].apply(lambda x: save_from_dataframe(x, path=valid_path))
    test_df['filename'].apply(lambda x: save_from_dataframe(x, path=test_path))
