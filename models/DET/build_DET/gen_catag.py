import obspy
import glob
import sys
import time
import os

from obspy.signal.trigger import recursive_sta_lta, trigger_onset
from obspy.signal.trigger import plot_trigger

folder_data = 'data/'
stations = ['SLRZ']
print(stations)

for i1, station in enumerate(stations):
	print(i1, station, len(stations))
	files_Z = sorted(glob.glob(folder_data+station+'/*HNZ*.miniseed'))

	#Creating folder
	try:
	        os.makedirs('picked/'+station+'/')
	except Exception:
	        print('Folder attributes already exist')

	#Header
	#year,jday,from_hh,from_mm,from_ss,to_hh,to_mm,to_ss
	f = open('picked/'+station+'/'+station+'.csv', 'w')
	f.write('year,jday,from_hh,from_mm,from_ss,to_hh,to_mm,to_ss'+'\n')
	f.close()

	c = 0
	for index_file, file_z in enumerate(files_Z):
		print(index_file, len(files_Z))
		st = obspy.read(file_z)
		flag_trigger = False

		#if st[0].stats.station == 'IQT0':
		if index_file >= 0:
			print(st)

			for i in range(0, len(st)):
				Fs = st[i].stats.sampling_rate
				if len(st[i].data) >= Fs*1800:
					st[i].detrend('demean') #remove mean
					st[i].detrend('linear') #trend linear
					#st[i].taper(0.05, 'cosine', side='both')
					st[i].filter('bandpass', freqmin=1.0, freqmax=7.0)

					cft = recursive_sta_lta(st[i].data, int(60*Fs), int(240*Fs))
					#plot_trigger(st[i], cft, 2.8, 0.5)
					#plt.show()
					#sys.exit()

					#2021,114,5,2,20,5,4,0 #Format catalog
					on_pb = 2.8
					on_of = trigger_onset(cft, on_pb, 0.5)

					if len(on_of) == 0:
						flag_write_st = False
					else:
						flag_write_st = True

					if flag_write_st == True:
						for j in range(0, on_of.shape[0]):
							time_on = st[i].stats.starttime+on_of[j][0]/Fs
							time_of = st[i].stats.starttime+on_of[j][1]/Fs

							if time_of - time_on < 600:
								flag_trigger = True
								time_on = time_on-60
								time_of = time_of+60

								f = open('picked/'+station+'/'+station+'.csv', 'a')
								f.write(str(time_on.year)+','+str(time_on.julday)+','+str(time_on.hour)+','+str(time_on.minute)
									+','+str(time_on.second)+','+str(time_of.hour)+','+str(time_of.minute)+','+str(time_of.second)+'\n')
								f.close()

					if flag_trigger == False:
						f = open('picked/'+station+'/'+station+'.csv', 'a')
						f.write(str(st[i].stats.starttime.year)+','+str(st[i].stats.starttime.julday)+',12,0,0,12,1,0\n')
						f.close()

