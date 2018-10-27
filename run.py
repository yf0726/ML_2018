# import basic packages
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

# import self-defined modules
from implementations import *
from tools_xia import *
from helpers_xia import *

# just to ingore warning
import warnings
warnings.filterwarnings('ignore')

data_path_tr = 'data/train.csv'
yb_tr, data_tr, idx_tr, labels = load_csv_data(data_path_tr, sub_sample=False)

# prepare the data using self-defined DataFrame class
labels_dataframe = ['Prediction'] + labels
data_tr_dataframe = np.concatenate((yb_tr.reshape([-1,1]), data_tr), axis=1)
dataframe_tr = DataFrame(data_tr_dataframe, idx_tr.tolist(), labels_dataframe)

# replace -999（undefined values of 'DER_mass_MMC'） with mode respectively in signal and background
temp = df_tr.loc(df_tr['DER_mass_MMC']==-999)
temp = temp.loc(temp['Prediction']==1)
df_tr.loc(temp.index)['DER_mass_MMC'] = 119.89

temp = df_tr.loc(df_tr['DER_mass_MMC']==-999)
temp = temp.loc(temp['Prediction']==-1)
df_tr.loc(temp.index)['DER_mass_MMC'] = 96.819

feature_dorp_phi = ['PRI_jet_leading_phi',
                    'PRI_jet_subleading_phi',
                    'PRI_lep_phi',
                    'PRI_met_phi',
                    'PRI_tau_phi',]
df_tr = df_tr.drop(feature_dorp_phi)

def group_features_by_jet(dataframe):
    return {  
        0: dataframe.loc( dataframe['PRI_jet_num'] == 0).copy(),
        1: dataframe.loc( dataframe['PRI_jet_num'] == 1).copy(),
        2: dataframe.loc((dataframe['PRI_jet_num'] == 2) | (dataframe['PRI_jet_num'] == 3)).copy(),
    }
df_tr_grp = group_features_by_jet(df_tr)

# get features with undefined values
df_tr_feature_undefined  = [['DER_deltaeta_jet_jet','DER_mass_jet_jet','DER_prodeta_jet_jet','DER_lep_eta_centrality','PRI_jet_leading_pt','PRI_jet_leading_eta','PRI_jet_subleading_pt','PRI_jet_subleading_eta'],['DER_deltaeta_jet_jet','DER_mass_jet_jet','DER_prodeta_jet_jet','DER_lep_eta_centrality','PRI_jet_subleading_pt','PRI_jet_subleading_eta'],[]]
    
# drop features with undefined values in each group
df_tr_grp[0] = df_tr_grp[0].drop(df_tr_feature_undefined[0])
df_tr_grp[1] = df_tr_grp[1].drop(df_tr_feature_undefined[1])
# group2 have no feature with undefined values so we donot need to drop it 

dataframe_tr_feature_zero = []
for i in range(len(dataframe_tr_grp)):
    dataframe_tr_miss = missing_rate(dataframe_tr_grp[i], 0)
    dataframe_tr_feature_zero.append(np.array(dataframe_tr_miss.labels)[dataframe_tr_miss.values == 1].tolist())

# drop features with undefined values in each group
# Because only the first group has all zeros column other than 'PRI_jet_num'
# here we only need to drop first group with dataframe_tr_feature_zero
dataframe_tr_grp[0] = dataframe_tr_grp[0].drop(dataframe_tr_feature_zero[0])

# drop feature 'PRI_jet_num' which is already used for grouping
dataframe_tr_grp[1] = dataframe_tr_grp[1].drop('PRI_jet_num')
dataframe_tr_grp[2] = dataframe_tr_grp[2].drop('PRI_jet_num')


tx = dataframe_tr_grp[i].drop('Prediction').values
y = (dataframe_tr_grp[i])['Prediction']


