# Developed by: Pablo E. Lara
# E3WS: Earthquake Early Warning starting from 3 seconds of records on a single station with machine learning
# Module to generate feature vector noise

import sys
import os
import datetime
import obspy
import glob
import numpy as np
import pandas as pd
import scipy as sc
import scipy.signal
from obspy.core.stream import Stream
from obspy.io.sac.sacpz import attach_paz

#MFCC
from python_speech_features import mfcc

#Features
import functions.pb_functions as pbi
import functions.pb_utils_v16 as pb_utils

#Set init time
init_time_ai = datetime.datetime.now()

#Write file
flag_writefile = 1

#Set station
G1 = ['SLRZ']
pb_inst = True #For using pole and zero

for ns_stat in G1:
	noise_station = ns_stat
	noise_loc = '00'
	high_corner_freq = 7

	folder_out = 'atr_noise/'+noise_station+'/'
	try:
		os.makedirs(folder_out)
	except Exception:
		print('Folder attributes already exist')

	#Generate earthquake tensor
	n_channels = 3
	count=0

	saspe_path = 'data/'
	picked_file = 'picked/'+noise_station+'/'+noise_station+'.csv'

	df = pd.read_csv(picked_file, index_col=False)
	print(df)

	pzfile = 'pz/'+noise_station+'_HN.pz'

	df_group = df.groupby(['year', 'jday']).size().rename('count').reset_index()
	print(df_group.shape)

	pz_file = open(pzfile, 'r')

	if flag_writefile == 1:
		print('Model will saved')
		model_characteristic = 'pb_slidding1s_'+str(df_group['year'].iloc[0])+str('{:03d}'.format(df_group['jday'].iloc[0]))+'to'+str(df_group['year'].iloc[-1])+str('{:03d}'.format(df_group['jday'].iloc[-1]))+'_NOISE_'+str(noise_station)+'_f'+str(high_corner_freq)+'_goodtrim_v16' #data/RDND/RDND/AS.00.RDND.HNE_20210422.miniseed
		print(model_characteristic)

	for i in range(0, df_group.shape[0]):
		jday = df_group['jday'].iloc[i]
		year = df_group['year'].iloc[i]

		df_day = df[(df['year']==year) & (df['jday']==jday)]
		print(df_day)

		t0_start = obspy.UTCDateTime(year=year, julday=jday, hour=0, minute=0, second=0)
		t0_end = obspy.UTCDateTime(year=year, julday=jday, hour=23, minute=59, second=59)

		noise_df = pd.DataFrame(columns=['from', 'to'])
		for i1 in range(0, df_day.shape[0]):
			from_hh = df_day['from_hh'].iloc[i1]
			from_mm = df_day['from_mm'].iloc[i1]
			from_ss = df_day['from_ss'].iloc[i1]

			to_hh = df_day['to_hh'].iloc[i1]
			to_mm = df_day['to_mm'].iloc[i1]
			to_ss = df_day['to_ss'].iloc[i1]

			if i1 == 0:
				from_eq = obspy.UTCDateTime(year=year, julday=jday, hour=from_hh, minute=from_mm, second=from_ss)
				to_eq = obspy.UTCDateTime(year=year, julday=jday, hour=to_hh, minute=to_mm, second=to_ss)
				noise_df = noise_df.append({'from':t0_start, 'to':from_eq-60}, ignore_index=True)

				if df_day.shape[0] != 1:
					from_hh_p1 = df_day['from_hh'].iloc[i1+1]
					from_mm_p1 = df_day['from_mm'].iloc[i1+1]
					from_ss_p1 = df_day['from_ss'].iloc[i1+1]
					t_end = obspy.UTCDateTime(year=year, julday=jday, hour=from_hh_p1, minute=from_mm_p1, second=from_ss_p1)-60
				else:
					t_end = t0_end
				noise_df = noise_df.append({'from':to_eq+60, 'to':t_end}, ignore_index=True)

			elif i1 < df_day.shape[0]-1 and i1 > 0:
				from_hh_p1 = df_day['from_hh'].iloc[i1+1]
				from_mm_p1 = df_day['from_mm'].iloc[i1+1]
				from_ss_p1 = df_day['from_ss'].iloc[i1+1]
				t_end = obspy.UTCDateTime(year=year, julday=jday, hour=from_hh_p1, minute=from_mm_p1, second=from_ss_p1)-60
				to_eq = obspy.UTCDateTime(year=year, julday=jday, hour=to_hh, minute=to_mm, second=to_ss)
				noise_df = noise_df.append({'from':to_eq+60, 'to':t_end}, ignore_index=True)

			else:
				to_eq = obspy.UTCDateTime(year=year, julday=jday, hour=to_hh, minute=to_mm, second=to_ss)
				noise_df = noise_df.append({'from':to_eq+60, 'to':t0_end}, ignore_index=True)

		print('Noise df:')
		noise_df = noise_df[noise_df['from'] < noise_df['to']]
		print(noise_df)

		for i1 in range(0, noise_df.shape[0]):
			jday = noise_df['from'].iloc[i1].julday
			t1 = noise_df['from'].iloc[i1]
			t2 = noise_df['to'].iloc[i1]

			year_ns = noise_df['from'].iloc[i1].year
			month_ns = noise_df['from'].iloc[i1].month
			day_ns = noise_df['from'].iloc[i1].day

			#print('{:02d}'.format(month_ns))

			pb_file = glob.glob(saspe_path+noise_station+'/*HNZ_'+str(year_ns)+str('{:02d}'.format(month_ns))+str('{:02d}'.format(day_ns))+'*.miniseed')[0]

			fp = pb_file.replace('HNZ', 'HNE')
			st_raw = obspy.read(fp, format='MSEED')
			fp = pb_file.replace('HNZ', 'HNN')
			st_raw += obspy.read(fp, format='MSEED')
			st_raw += obspy.read(pb_file, format='MSEED') #HNZ already in files[i2]

			st_process = st_raw.copy()
			st_process.trim(t1, t2)

			st_control = st_process.copy().select(component="Z")

			for ip in range(0, len(st_control)):
				Fs = st_control[ip].stats.sampling_rate
				if len(st_control[ip].data) >= Fs*1800: #Just process traces if is greater than 30m
					slidding_seconds = 1
					N = len(st_control[ip].data)
					w_sec = 10
					n_waves = int(N/(st_control[ip].stats.sampling_rate)-w_sec+1) #slidding 1s

					for i3 in range(0, n_waves):
						st = st_process.copy()

						t1 = i3
						t2 = w_sec+i3
						st.trim(st_control[ip].stats.starttime+t1, st_control[ip].stats.starttime+t2)
						print(i1, ip, t1, t2, n_waves)

						#Quality control
						num_points = int(w_sec*st[0].stats.sampling_rate)+1
						try:
							E_NM = np.abs(st[0].data-np.mean(st[0].data)).sum()
							N_NM = np.abs(st[1].data-np.mean(st[1].data)).sum()
							Z_NM = np.abs(st[2].data-np.mean(st[2].data)).sum()
							flag_NM = E_NM != 0 and N_NM != 0 and Z_NM!=0
							flag_num_points = len(st[0].data) == num_points  and len(st[1].data) == num_points  and len(st[2].data) == num_points
						except Exception:
							flag_NM = False
							flag_num_points = False

						if flag_NM == True and flag_num_points == True:
							Fs = st[0].stats.sampling_rate #Initial Fs

							#Just apply instrumental correction for SASPE, STEAD has already done this part
							if pb_inst == True:
								pzfile = pzfile
							else:
								pzfile = None

							FV = pb_utils.st_FV(st, pb_inst, pzfile=pzfile, fmin=1.0, fmax=high_corner_freq)

							if flag_writefile == 1:
								f=open(folder_out+'/DET_'+str(w_sec)+'s_'+model_characteristic+'.csv','a')
								#f.write(str(event_id)+','+ str(e2_date)+',')
								for i3 in range(0, len(FV)):
									f.write(str(FV[i3])+',')
								f.write('SASPe'+','+noise_station+',')
								f.write(str(0)+'\n') #0=NOISE, #1=EARTHQUAKE
								f.close()

							count = count+1
						st.clear()

#Show time complexity
final_time_ai = datetime.datetime.now()
print(final_time_ai - init_time_ai)
