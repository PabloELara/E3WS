import numpy as np
import pandas as pd
import datetime
import joblib
import sys
import os
import glob

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

print('Start Machine Learning seismic discrimination')
init_time_ai = datetime.datetime.now()

np.random.seed(30002)

###### Global variables ######
pb_version = 16
high_corner_freq = 7
tr_ptime = np.arange(8)*0.5+0.5
window = 10
target = 'DET'
station = 'SLRZ'
samples_noise = 900000
db = 'Whole' #Used for training, STEAD, Peru, Chile, Japan or Whole
pb_model = 'XGB'
name_eq = 'Whole' #training_balanced, training_unbalanced, Whole
out_target = 'ACC'
add_zeros = False

#Print version
print('VERSION:', str(pb_version))

if pb_version == 16:
        n_attr = 172 #Total num of attributes in attr file

#Creating folder
pb_folder = 'saved_models/'
print('Folder', pb_folder)
file_pb = target+'_'+station+'_'+db+'_'+pb_model+'_'+str(window)+'tp'+str(np.min(tr_ptime))+'to'+str(np.max(tr_ptime))+'each0.5f'+str(high_corner_freq)+'_unbalanced_official_v'+str(pb_version)
fname = pb_folder+file_pb
print(fname)

#Creating folder
try:
        os.makedirs(pb_folder)
except Exception:
        print('Folder attributes already exist')

#Function to convert label to numeric
def pb_label2num(t, label):
        center = 0.5
        if t >= center:
                if label == 'P':
                        s = 1
                else:
                        s = 2
        else:
                s = 0
        if label == 'N':
                s = 0
        return s

def P_pick_y2value_DET(y):
        label = y[0]
        slide = float(y[1:].replace('+-', '-'))
        slide = pb_label2num(slide, label)
        return slide

#Read feature names
df_col = pd.read_csv('pb_index_DET_v'+str(pb_version)+'.csv')

#Deleting columns that we are not use (old features)
DEL_COL = ['lambda_2', 'lambda_3', 'E_maxturnpoints_t', 'N_maxturnpoints_t', 'Z_maxturnpoints_t',
'E_IncDec_t', 'N_IncDec_t', 'Z_IncDec_t',
'E_growth_t', 'N_growth_t', 'Z_growth_t',
'E_minturnpoints_t', 'N_minturnpoints_t', 'Z_minturnpoints_t',
'E_maxturnpoints_f', 'N_maxturnpoints_f', 'Z_maxturnpoints_f',
'E_minturnpoints_f', 'N_minturnpoints_f', 'Z_minturnpoints_f',
'cov_EN', 'cov_EZ', 'cov_NZ',
'E_Erms_t', 'N_Erms_t', 'Z_Erms_t',
'E_Etot_f', 'N_Etot_f', 'Z_Etot_f',
'E_Erms_f', 'N_Erms_f', 'Z_Erms_f']

#Folder eq.
folder_eq = 'atr_eq/'

#Eq.e files
files_eq = [folder_eq+'DET_'+str(format(window-tr_pt, '.2f'))+'tp'+str(format(tr_pt, '.2f'))+'_pb_Whole_mag3.0dis200.0dep100.0f7_withindex_PSlabel_goodtrim_v'+str(pb_version)+'.csv' for tr_pt in tr_ptime]
print(files_eq)

#Eq. dataframes
df_eq = pd.concat([pd.read_csv(f, header=None) for f in files_eq])
df_eq.columns = df_col.feature.values
df_eq = df_eq.reset_index(drop=True)
print(df_eq)

#Delete some columns
df_eq = df_eq.drop(columns=DEL_COL)
df_eq = df_eq.reset_index(drop=True)

#Remove ENZ_same
df_eq = df_eq[df_eq['ENZ_equal'] == 'ENZ_different']

#Noise atr
df_col_saspe = pd.read_csv('pb_index_noise_v'+str(pb_version)+'.csv') #140 attributes+pb_inst,station,label

#Noise. dataframes
FILE_ns = sorted(glob.glob('atr_noise/'+station+'/*'))

for i1, file_ns in enumerate(FILE_ns):
	print(i1)
	df_ns0 = pd.read_csv(file_ns, header=None)
	print(df_ns0)
	if i1 == 0:
		df_ns = df_ns0
	else:
		df_ns = pd.concat([df_ns, df_ns0])

df_ns.columns = df_col_saspe.feature.values
df_ns = df_ns.reset_index(drop=True)
print(df_ns)

try:
	df_ns = df_ns.sample(samples_noise)
except Exception:
	print('The number of noise samples are less than '+str(samples_noise)+', please place more samples.')
	sys.exit()
df_ns = df_ns.reset_index(drop=True)

#Set initial df
df = df_eq

#Convert target to integer class N=0, P=1, S=2
df['y_label'] = df['PSaft'].apply(lambda y: P_pick_y2value_DET(y)) #Convert y to function
print('Total dataframe with numerical label:')
print(df)

#df_train = df[df['stage']=='Training']
#df_test = df[df['stage']=='Testing']

#Train all the samples
df['stage'] = 'Training'

#Concatenate eq and noise df
df_tot = pd.concat([df_eq, df_ns])
df_train = df_tot
print('Total df')
print(df_train)

#Set X_train
x_train = df_train.iloc[:, 0:n_attr-len(DEL_COL)]
print('X_train attributes:')
print(x_train)

for i,col in enumerate(x_train.columns):
	print(i, col)

#Set Y_train
y_train = df_train[['y_label']]
print('Y_train label:')
print(y_train)

model_xgb = make_pipeline(StandardScaler(), xgb.XGBClassifier(colsample_bytree=0.4603, gamma=0.0468,
  learning_rate=0.05, max_depth=4,
  min_child_weight=1.7817, n_estimators=6000,
  reg_alpha=0.4640, reg_lambda=0.8571,
  subsample=0.8, verbosity=0,
  nthread = -1))

#Define model
if pb_model == 'XGB':
        model = model_xgb

x_train = np.array(x_train)
y_train = np.array(y_train)

print('Train shape:', x_train.shape, y_train.shape)
#sys.exit()

#Train model
model.fit(np.array(x_train), np.array(y_train))

#Saving model
joblib.dump(model, open(fname+'.joblib', 'wb'), compress=3)

#Show time complexity
final_time_ai = datetime.datetime.now()
print(final_time_ai - init_time_ai)
