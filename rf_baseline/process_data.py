import pandas as pd
import ujson as json
import numpy as np
import os
from tqdm import tqdm
import subprocess
import FATS
from scipy.optimize import fmin_l_bfgs_b


meta_path='../data/clean-eros/'
metadict = {'test':'../data/clean-eros/test/',
            'train':'../data/clean-eros/train/',
            'valid':'../data/clean-eros/valid/'}

freq_feat_list = []
for i in range(1, 4):
    for j in range(4):
        freq_feat_list.append('Freq'+str(i)+'_harmonics_amplitude_'+str(j))
        freq_feat_list.append('Freq'+str(i)+'_harmonics_rel_phase_'+str(j))



def merge_two_dicts(x, y):
    """Given two dicts, merge them into a new dict as a shallow copy."""
    z = x.copy()
    z.update(y)
    return z

def listdir_nohidden(path):
    list_files = []
    for f in os.listdir(path):
        if not f.startswith('.'):
            list_files.append(f)
    return list_files

def spectral_kernel_matrix(dx, q=1, parameters=[1.0, 1.0, 1.0], diff=0):
    if q < 1:
        raise ValueError("At least one component needed")
    if not len(parameters) == 3*q:
        print(len(parameters))
        print(q)
        raise ValueError("Missing parameters!")
    w = parameters[0::3]
    v = parameters[1::3]
    mu = parameters[2::3]
    if diff == 0:
        K = np.zeros(shape=dx.shape)
        for k in range(0, q):
            K += w[k]*np.multiply(np.exp(-2.0*np.pi**2*np.power(dx, 2.0)*v[k]), np.cos(2.0*np.pi*dx*mu[k]))
        return K
    elif np.mod(diff-1, 3) == 0: # weight
        k = int((diff-1)/3)
        return np.multiply(np.exp(-2.0*np.pi**2*np.power(dx, 2.0)*v[k]), np.cos(2.0*np.pi*dx*mu[k]))
    elif np.mod(diff-2, 3) == 0: # variance
        k = int((diff-2)/3)
        return w[k]*np.multiply(np.multiply(np.exp(-2.0*np.pi**2*np.power(dx, 2.0)*v[k]), np.cos(2.0*np.pi*dx*mu[k])), -2.0*np.pi**2*np.power(dx, 2.0))
    elif np.mod(diff-3, 3) == 0: # mean
        k = int((diff-3)/3)
        return w[k]*np.multiply(np.multiply(np.exp(-2.0*np.pi**2*np.power(dx, 2.0)*v[k]), np.sin(2.0*np.pi*dx*mu[k])), -2.0*np.pi*dx)

def gp_regression_spectral(x, y, dy, xp=None, q=1, parameters=[1.0, 1.0, 1.0], debug=False):
    N = len(x)
    if xp is None:
        Np = 100
        xp = np.linspace(x[0], x[N-1], num=Np)
    else:
        Np = len(xp)

    dx = np.tile(x, (N, 1))
    dxpp = np.tile(xp, (Np, 1))
    dxp = np.tile(xp, (N, 1)).T - np.tile(x, (Np, 1))
    K = spectral_kernel_matrix(dx-dx.T, parameters=parameters, q=q)
    Kpp = spectral_kernel_matrix(dxpp-dxpp.T, parameters=parameters, q=q)
    Kp = spectral_kernel_matrix(dxp, parameters=parameters, q=q)
    invK = np.linalg.inv(K + np.diag(np.power(dy, 2.0)))
    yp = np.dot(Kp, np.dot(invK, y))
    varyp = Kpp - np.dot(Kp, np.dot(invK, Kp.T))
    varyp +=  np.median(np.power(dy, 2.0))*np.eye(Np)
    varyp = np.diag(varyp)

    if debug == True:
        fig = plt.figure(figsize=(6, 3), dpi=80)
        ax = fig.add_subplot(1, 1, 1)
        ax.errorbar(x, y, dy, color='b', fmt='.', linewidth=1)
        ax.plot(xp, yp, color='g', linewidth=2)
        ax.fill_between(xp, yp - np.sqrt(varyp), yp + np.sqrt(varyp), facecolor='g', alpha=0.2)
        ax.set_ylabel('Normalized flux')
        ax.set_xlabel('MJD - t0')
        ax.set_title("Spectral kernel q: "+str(q))
        plt.grid()
    return xp, yp, varyp



def buildFeature(path, value):
    test_files = listdir_nohidden(path)
    y = []
    for file_idx in tqdm(range(len(test_files))):
        file_name = test_files[file_idx]
        with open(path+file_name, 'r') as fp:
            datadict = json.load(fp)
        xpg, ypg, varypg = gp_regression_spectral(np.array(datadict['time_g']),
                                                  np.array(datadict['data_g']),
                                                  np.array(datadict['data_err_g']))
        xpr, ypr, varypr = gp_regression_spectral(np.array(datadict['time_r']),
                                                  np.array(datadict['data_r']),
                                                  np.array(datadict['data_err_r']))
        period = datadict['period']
        feature_learner = FATS.FeatureSpace(Data=['magnitude','time', 'error'], excludeList = ['Amplitude', 'FluxPercentileRatioMid20',
                                                                        'FluxPercentileRatioMid35',
                                                                        'FluxPercentileRatioMid50',
                                                                        'FluxPercentileRatioMid65',
                                                                        'FluxPercentileRatioMid80',
                                                                        'PercentDifferenceFluxPercentile',
                                                                        'PeriodLS', 'Period_fit',
                                                                        'Psi_CS', 'Psi_eta']+freq_feat_list)
        features_g = feature_learner.calculateFeature(np.array([ypg, xpg, varypg]))
        features_g = features_g.result(method='dict')
        features_g = {k+'_g': v for k, v in features_g.items() if v}
        features_r = feature_learner.calculateFeature(np.array([ypr, xpr, varypr]))
        features_r = features_r.result(method='dict')
        features_r = {k+'_r': v for k, v in features_r.items() if v}
        features = merge_two_dicts(features_g, features_r)
        features['period'] = period
        if file_idx == 0:
            res_df = pd.DataFrame(columns=features.keys())
            res_df = res_df.append(features, ignore_index=True)
        else:
            res_df = res_df.append(features, ignore_index=True)
        y.append(datadict['class_array'])
    y = np.array(y)
    #Saving the response array
    np.save(meta_path+value+'.npy', y)
    res_df.to_csv(meta_path+value+'.csv', index=False)
    print "Processing for %s data completed"%(value)


def process_data():
    """
    for key in metadict.keys():
        buildFeature(metadict[key], key)
    """
    key = 'test'
    buildFeature(metadict[key], key)

if __name__ == '__main__':
    process_data()
