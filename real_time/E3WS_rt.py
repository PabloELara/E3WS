# -*-coding:Utf-8 -*
#E3WS
#Simple simulate real-time from MSEED files

#Dependences
import obspy
import numpy as np
import matplotlib.pyplot as plt
import joblib
import collections
import os
import sys
import datetime

#Features
import functions.pb_functions as pbi
import functions.pb_utils_v16 as pb_utils
import functions.pick_func_v16 as pick_func
from functions.utils import pb_SAM

#Software parameters
flag_plot = 1 #plot detection, P-phase picking and first magnitude estimation
flag_writefile = 1 #write logs
flag_trackmag = True
pb_version = 16
pb_subversion = 0
n_models = 5 #number of models for tracking source characterization, 1: just 3 sec model, 58: 3 sec to 60 sec models
pb_inst = True #True for instrument correction, you have to add pz/ folder with pzfile

#Model parameters
Ptrain = 0.5 #Dont change
mov_time = 0.1 #moving time for estimate P-phase arrival time
thr = 0.80 #detection threshold to trigger an event
folder_models = '../models/'

#Station parameters
station = 'SLRZ'
sta_lat = -12.074200
sta_lon = -77.233200

if pb_inst == True:
	print('Using Pole Zero')
	pzfile = 'pz/'+station+'_HN.pz'
else:
	print('NOT using Pole Zero')
	pzfile = None

print('PZFILE:', pzfile)

#Folder out
folder_out = 'results'

#Earthquake name
eq_name = 'Canta_M5.6_20220107'

#For BAZ models
if n_models > 3:
	n_models_baz = 3
else:
	n_models_baz = n_models

#Creating folder
if flag_writefile == 1:
	try:
		os.makedirs(folder_out)
	except Exception:
		print('Folder out already exist')

	#Write headers
	f=open(folder_out+'/'+station+'_'+eq_name+'.csv','w')
	f.write('P_AI_date,starttime,endtime,P-wave(s),mag_pred(M),lat_pred,lon_pred,dis_pred(km),dep_pred(km),baz_pred(°)\n')
	f.close()

#Reference in time
REF = np.array([])

#Waveform file
#org_time = obspy.UTCDateTime('2022-01-07T10:27:06.30')#By IGP, just for trim
file_E = station+'.HNE.PE.2022.007.mseed'
file_N = station+'.HNN.PE.2022.007.mseed'
file_Z = station+'.HNZ.PE.2022.007.mseed'
format = 'MSEED'

#Read file
data = 'data/'+eq_name+'/'
st_raw = obspy.read(data+file_E, format=format)
st_raw += obspy.read(data+file_N, format=format)
st_raw += obspy.read(data+file_Z, format=format)

print('Reading models')
if flag_trackmag == 1:
	#n_models = 1 #58 in total, 1 model, 1 estimation
	MAG_md = ['MAG_Whole_StackingXGB_7tp'+str(i)+'f45_v'+str(pb_version)+'.'+str(pb_subversion)+'.joblib' for i in np.arange(n_models)+3]
	DIS_md = ['DIS_Whole_StackingXGB_7tp'+str(i)+'f45_v'+str(pb_version)+'.'+str(pb_subversion)+'.joblib' for i in np.arange(n_models)+3]
	DEP_md = ['DEP_Whole_StackingXGB_7tp'+str(i)+'f45_v'+str(pb_version)+'.'+str(pb_subversion)+'.joblib' for i in np.arange(n_models)+3]
	BAZ_cos_md = ['welloriented_BAZ_Cos_STEAD_StackingXGB_7tp'+str(i)+'f45_v'+str(pb_version)+'.'+str(pb_subversion)+'.joblib' for i in np.arange(n_models_baz)+3]
	BAZ_sin_md = ['welloriented_BAZ_Sin_STEAD_StackingXGB_7tp'+str(i)+'f45_v'+str(pb_version)+'.'+str(pb_subversion)+'.joblib' for i in np.arange(n_models_baz)+3]

	#print files:
	print(MAG_md)

	MAG_pb = [joblib.load(open(folder_models+'MAG/'+MAG_md[i], 'rb')) for i in range(n_models)]
	DIS_pb = [joblib.load(open(folder_models+'DIS/'+DIS_md[i], 'rb')) for i in range(n_models)]
	DEP_pb = [joblib.load(open(folder_models+'DEP/'+DEP_md[i], 'rb')) for i in range(n_models)]
	BAZ_cos_pb = [joblib.load(open(folder_models+'BAZ/'+BAZ_cos_md[i], 'rb')) for i in range(n_models_baz)]
	BAZ_sin_pb = [joblib.load(open(folder_models+'BAZ/'+BAZ_sin_md[i], 'rb')) for i in range(n_models_baz)]


#Reading DET and PICK model
DET_pb = joblib.load(open('../models/DET/DET_'+station+'_Whole_XGB_10tp0.5to4.0each0.5f7_unbalanced_official_v16.joblib', 'rb'))
PICK_pb = joblib.load(open('../models/PICK/PICK_CLF_XGBdep4n6000bytree0.46level1.0node1.0_P0.5_NPS_eachpoint_spike_v16.joblib', 'rb'))

print('Reading models complete')

#Print raw data
print(st_raw)

#Select ENZ 
st_e = st_raw.select(component='E')
st_n = st_raw.select(component='N')
st_z = st_raw.select(component='Z')

#Quality control of traces
n_traces = min(len(st_e), len(st_n), len(st_z))

for i in range(0, n_traces):
	print(i, n_traces)
	st = obspy.Stream()
	st = st.append(st_e[i])
	st = st.append(st_n[i])
	st = st.append(st_z[i])
	maxstart = np.max([tr.stats.starttime for tr in st])
	minend =  np.min([tr.stats.endtime for tr in st])
	st.trim(maxstart, minend)

	st_process = st.copy()

	Fs = st_process[0].stats.sampling_rate
	slidding_seconds = 1
	N = len(st_process[0].data)
	w_sec = 10

	n_waves = int((N/Fs-w_sec)/slidding_seconds)+1 #slidding 1s

	PROB_P = np.array([])
	PROB_N = np.array([])
	PROB_S = np.array([])
	PROB_PP = np.array([])
	buffer_proba = collections.deque(maxlen=5)
	buffer_proba_n = collections.deque(maxlen=5)
	eq_thr_past = 0
	eq_thr_present = 0

	flag_magnitude = 0
	c_mag = 0 #counter magnitude (1 times)
	waiting_sec = 0
	n_eq = 0
	DL = np.array([])
	for i3 in range(0, n_waves):
		st_pb = st_process.copy()
		t1 = i3*slidding_seconds
		t2 = w_sec+t1
		st_pb = st_pb.trim(st_pb[0].stats.starttime+t1, st_pb[0].stats.starttime+t2)

		#Feature vector
		FV = pb_utils.st_FV(st_pb, pb_inst, pzfile=pzfile, fmin=1.0, fmax=7.0)
		FV = np.real([FV])

		#Get probabilities
		prob_nps = DET_pb.predict_proba(FV)[0]
		prob_n = prob_nps[0] #v8
		prob_p = prob_nps[1] #v8
		prob_s = prob_nps[2] #v8
		print(i3, round(t2,3), st_pb[0].stats.starttime, st_pb[0].stats.endtime, len(st_pb[0].data), prob_n, prob_p, prob_s, n_waves)

		PROB_N = np.append(PROB_N, prob_n)
		PROB_P = np.append(PROB_P, prob_p)
		PROB_S = np.append(PROB_S, prob_s)

		#Circular vector probability
		buffer_proba.append(prob_p)
		buffer_proba_np = np.array(buffer_proba)
		buffer_proba_np = np.flip(buffer_proba_np) #First element is the current proba

		buffer_proba_n.append(prob_n) #Prob of being N
		buffer_proba_of = np.array(buffer_proba_n)
		buffer_proba_of = np.flip(buffer_proba_of) #First element is the current proba

		#Logic for trigger
		if np.mean(buffer_proba_np[0:3])>= thr and len(buffer_proba_np) >= 3:
			eq_thr_present = 1

		if eq_thr_present == 1:
			if np.mean(buffer_proba_of) >= 0.99:
				eq_thr_present = 0

		if eq_thr_past == 0 and eq_thr_present == 1:
			print('New event!:')

			w_sec_pick = 8
			st_pick = st_pb.copy().trim(st_pb[0].stats.endtime-w_sec_pick, st_pb[0].stats.endtime)

			PROB_PP = pick_func.PP_pick(st_pick, mov_time, pb_inst, pzfile, PICK_pb, fmin=1.0, fmax=7.0) #SASPe for use pzfile, STEAD for not
			P_AI_date = st_pick[0].stats.starttime+np.argmax(PROB_PP)*mov_time+4-Ptrain
			print('P arrival at:', P_AI_date) #Must delete

			if flag_plot == 1:
				REF = np.append(REF, st_pick[0].stats.starttime-st[0].stats.starttime)
				if len(REF) == 1:
					PROB_PP_TENSOR = np.array([PROB_PP])
					P_AI_DATE_VECTOR = [str(P_AI_date)]
				else:
					PROB_PP_TENSOR = np.vstack((PROB_PP_TENSOR, PROB_PP))
					P_AI_DATE_VECTOR.append(str(P_AI_date))

			#Calculate magnitude
			flag_magnitude = 1 #Change 1 to calculate source parameters

		if flag_magnitude == 1:
			print('MAG_time:', st_pb[0].stats.endtime-P_AI_date)
			#Wait until trace has minimum 3.0 sec
			if st_pb[0].stats.endtime-P_AI_date >= 3.0:

				st_pb_reg = st_process.copy().trim(P_AI_date-7, P_AI_date+3+c_mag)

				#FV for REG and BAZ
				FV45 = pb_utils.st_FV(st_pb_reg, pb_inst, pzfile=pzfile, fmin=1.0, fmax=45.0)
				FV45 = np.real([FV45])

				#Regression for MAG, DIS and DEP
				MAG_pb_predict = MAG_pb[c_mag].predict(FV45)[0]
				DIS_pb_predict = DIS_pb[c_mag].predict(FV45)[0]
				DEP_pb_predict = DEP_pb[c_mag].predict(FV45)[0]
				if c_mag <= n_models_baz-1:
					BAZ_pb_cos_predict = BAZ_cos_pb[c_mag].predict(FV45)[0]
					BAZ_pb_sin_predict = BAZ_sin_pb[c_mag].predict(FV45)[0]

				#Cos, Sin, to BAZ angle
				BAZ_pb_predict = (np.arctan2(BAZ_pb_sin_predict, BAZ_pb_cos_predict)*180/np.pi)

				#Get eq.lat, eq.lon
				eq_lat, eq_lon = pb_utils.pb_getpoint(sta_lat, sta_lon, DIS_pb_predict, BAZ_pb_predict)
				eq_lat = round(eq_lat, 6)
				eq_lon = round(eq_lon, 6)

				print('-->', P_AI_date, st_pb_reg[0].stats.starttime, st_pb_reg[0].stats.endtime, 
					st_pb_reg[0].stats.endtime-st_pb_reg[0].stats.starttime, MAG_pb_predict, eq_lat, eq_lon, 
					DIS_pb_predict, DEP_pb_predict)

				if flag_trackmag == True and flag_writefile == 1:
					f=open(folder_out+'/'+station+'_'+eq_name+'.csv','a')
					f.write(str(P_AI_date)+','+str(st_pb_reg[0].stats.starttime)+','+str(st_pb_reg[0].stats.endtime)+','+
						str(st_pb_reg[0].stats.endtime-st_pb_reg[0].stats.starttime-7)+','+str(MAG_pb_predict)+
						','+str(eq_lat)+','+str(eq_lon)+','+str(DIS_pb_predict)+','+str(DEP_pb_predict)+
						','+str(BAZ_pb_predict)+'\n')
					f.close()

				#Save first magnitude estimation just for plot
				if c_mag == 0:
					MAG_3s = MAG_pb_predict

				c_mag = c_mag+1

		#computing magnitude c_mag times
		if c_mag == n_models:
			flag_magnitude = 0
			c_mag = 0

		#Save past values
		eq_thr_past = eq_thr_present


#If we want to plot
if flag_plot == 1:
                #Plot picking
                time = np.arange(len(st[0].data))/Fs
                time_AI = np.arange(PROB_P.shape[0])+w_sec

                fig = plt.figure()
                ax = fig.add_subplot(311)
                plt.plot(time, st[2].data, color='black')
                for i in range(0, len(REF)):
                        plt.axvline(x=(obspy.UTCDateTime(P_AI_DATE_VECTOR[i])-st[0].stats.starttime), color='r', linestyle='--', label='AI P-arrival')
                plt.legend(loc='upper left')
                plt.title('E3WS, $\mathrm{Mag_{AI}}$: '+str(round(MAG_3s,1))+', P-phase used: 3s')
                plt.xlim([0, time[-1]])
                ax.axes.xaxis.set_ticklabels([])

                ax = fig.add_subplot(312)
                #PROB_P[0:14] = 0
                PROB_thr = np.convolve(PROB_P, np.ones(3), 'valid') / 3 #Plot average each 3 consecutive elements of PROB_P
                #plt.plot(time_AI, PROB_P, '*-', color='black')
                plt.plot(time_AI[2:], PROB_thr, '*-', color='black')
                plt.ylabel('P-phase prob.')
                plt.xlim([0, time[-1]])
                ax.axes.xaxis.set_ticklabels([])

                ax = fig.add_subplot(313)

                for i in range(0, len(REF)):
                        time_PP = np.arange(len(PROB_PP_TENSOR[i]))*mov_time+REF[i]+4
                        PROB_PP_plot = PROB_PP_TENSOR[i]/np.max(PROB_PP_TENSOR[i])
                        #plt.plot(time_PP, PROB_PP_plot, color='#1f77b4')
                        plt.plot(time_PP, PROB_PP_plot, color='black')
                        plt.xlabel('Time (s)')
                        plt.ylabel('P-arrival prob.')
                        plt.xlim([0, time[-1]])

                plt.show()

