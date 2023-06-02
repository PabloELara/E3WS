import numpy as np
import scipy as sc
import scipy.signal
#import utm
import obspy
import datetime
import matplotlib.pyplot as plt
from obspy.io.sac.sacpz import attach_paz
from pygeodesy.ellipsoidalVincenty import LatLon
import sys

#MFCC
from python_speech_features import mfcc

#Features
import functions.pb_functions as pbi


#Get feature vector from stream object
def st_FV(st, pb_inst, pzfile, fmin, fmax):
        FV = np.array([])
        st_cp = st.copy()
        Fs = st_cp[0].stats.sampling_rate
        #st_cp.plot()

        #Setting parameters and apply signal conditioning
        st_cp.detrend('demean') #remove mean
        st_cp.detrend('linear') #trend linear
        st_cp.taper(0.05, 'cosine', side='both')

        if Fs/2 >= fmax:
                st_cp.filter('bandpass', freqmin=fmin, freqmax=fmax, zerophase=True, corners=4) #If Fhigh-corner>Fs/2, obspy apply just highpass
        else:
                st_cp.filter('highpass', freq=fmin, zerophase=True, corners=4)

        Freq_pb = 100 #100Hz design
        if Fs != Freq_pb:
                st_cp.resample(Freq_pb)
        st_cp.detrend('demean') #Second offset elimination
        Fs = st_cp[0].stats.sampling_rate #Setting again, for new Fs

        if pb_inst == True:
                for i2 in range(0, len(st_cp)):
                        #Instrumental correction (in case using deconvolution)
                        attach_paz(st_cp[i2], pzfile , todisp=False, torad=False)
                        zeros = np.array(st_cp[i2].stats.paz.zeros)
                        zeros = np.delete(zeros, np.argwhere(zeros==0)[0:2]) #Remove 2 zeros to get acceleration from displacement
                        poles = np.array(st_cp[i2].stats.paz.poles)
                        constant = st_cp[i2].stats.paz.gain
                        sts2 = {'gain': constant, 'poles': poles, 'sensitivity': 1, 'zeros': zeros}
                        st_cp[i2].simulate(paz_remove=sts2) #SASPe

        #Second signal conditioning
        st_cp.detrend('demean') #remove mean
        st_cp.detrend('linear') #trend linear
        st_cp.taper(0.05, 'cosine', side='both')
        st_cp.filter('bandpass', freqmin=fmin, freqmax=fmax, zerophase=True, corners=4) #If Fhigh-corner>Fs/2, obspy apply just highpass

        #BAZ features
        data_baz = np.array([st_cp[0].data, st_cp[1].data, st_cp[2].data])

        cov = np.cov(data_baz)
        #FV = np.append(FV, cov[0,1])
        #FV = np.append(FV, cov[0,2])
        #FV = np.append(FV, cov[1,2])

        w, v = np.linalg.eig(cov)
        w_sorted = np.sort(w)
        FV = np.append(FV, w_sorted[-1])
        #FV = np.append(FV, w_sorted[-2])
        #FV = np.append(FV, w_sorted[-3])
        FV = np.append(FV, w_sorted[-1]/(w_sorted[-2]+w_sorted[-3]))
        FV = np.append(FV, v[:, np.argmax(w)])

        for i2 in range(0, len(st_cp)):
                #******* TEMPORAL DOMAIN *******#
                #Parameters:
                Fs = st_cp[i2].stats.sampling_rate
                N = st_cp[i2].stats.npts
                data_time = st_cp[i2].data
                time = np.linspace(0.0, (N-1)/Fs, N)

                #Hilbert to get envelope
                analytic_signal_t = sc.signal.hilbert(data_time)
                amp_env_t = np.abs(analytic_signal_t)

                #******* SPECTRAL DOMAIN *******#
                #Pre-processing PSD Welchs 75% overlapping
                N_fft = 1024
                per_overlap = 75
                if len(data_time) < N_fft:
                        N_fft=len(data_time)
                N_overlap=N_fft*per_overlap//100

                #Welch's PSD
                f, Pxx= sc.signal.welch(data_time, Fs, window='hanning', detrend=False, nperseg=N_fft, noverlap=N_overlap) #75% overlapping

                #Parameters
                data_spec = Pxx
                freq = f

                #******* MFCC ANALYSIS *******#
                #Extract 13 Melfs components
                mfcc_feat = mfcc(data_time, samplerate=Fs, winlen=N/Fs, winstep=N/Fs, nfft=N, numcep=13, nfilt=26, lowfreq=1.0, highfreq=Fs/2, appendEnergy=False)

                #Delta
                #d_mfcc_feat = delta(mfcc_feat, 2)

                #Delta-delta
                #dd_mfcc_feat = delta(d_mfcc_feat, 2)

                #Converting 1d array
                mfcc_feat = mfcc_feat.reshape(-1)
                #d_mfcc_feat = d_mfcc_feat.reshape(-1)
                #dd_mfcc_feat = dd_mfcc_feat.reshape(-1)

                #Domains
                Domain = ['Temp', 'Spec', 'Ceps']
                #Domain = ['Spec', 'Ceps']

                for domain in Domain:
                        #print('\nDomain:', domain)
                        #Temporal signal
                        if domain == 'Temp':
                                #Energy
                                FV = np.append(FV, pbi.maxEt(data_time)) #1
                                FV = np.append(FV, pbi.argmaxEt(data_time, time)) #2
                                FV = np.append(FV, pbi.centroid_t(data_time, time)) #3
                                FV = np.append(FV, pbi.BW_t(data_time, time)) #4
                                FV = np.append(FV, pbi.skewness_t(data_time, time)) #5
                                FV = np.append(FV, pbi.kurtosis_t(data_time, time)) #6
                                #FV = np.append(FV, np.max(pbi.diff_turn_points(data_time**2))) #7
                                #FV = np.append(FV, np.min(pbi.diff_turn_points(data_time**2))) #8
                                FV = np.append(FV, np.sum(data_time**2)) #9
                                #FV = np.append(FV, pbi.Erms(data_time**2)) #10

                                #Envelope
                                FV = np.append(FV, pbi.TCR_t(amp_env_t, 0.8, Fs)) #11
                                FV = np.append(FV, pbi.RMM(amp_env_t)) #12
                                FV = np.append(FV, np.mean(amp_env_t)) #13
                                FV = np.append(FV, np.std(amp_env_t)) #14
                                FV = np.append(FV, sc.stats.skew(amp_env_t)) #15
                                FV = np.append(FV, sc.stats.kurtosis(amp_env_t, fisher=False)) #16
                                #FV = np.append(FV, pbi.IncDec_env(amp_env_t,time)) #17
                                #FV = np.append(FV, pbi.Growth_env(amp_env_t, time)) #18
                                FV = np.append(FV, pbi.mTCR_t(amp_env_t, 0.8)) #19
                                FV = np.append(FV, pbi.shannon_ent(amp_env_t, 200)) #20
                                FV = np.append(FV, pbi.renyi_ent(amp_env_t,2, 200)) #21

                                #Original timeseries
                                FV = np.append(FV, pbi.ZCR_t(data_time, Fs))# 22

                        #Spectral signal (PSD)
                        if domain == 'Spec':
                                #PSD
                                FV = np.append(FV, np.max(data_spec)) #23
                                FV = np.append(FV, pbi.argmaxEf(data_spec, freq)) #24
                                FV = np.append(FV, pbi.centroid_f(data_spec ,freq)) #25
                                FV = np.append(FV, pbi.BW_f(data_spec, freq)) #26
                                FV = np.append(FV, pbi.skewness_f(data_spec, freq)) #27
                                FV = np.append(FV, pbi.kurtosis_f(data_spec, freq)) #28
                                #FV = np.append(FV, np.max(pbi.diff_turn_points(data_spec))) #29
                                #FV = np.append(FV, np.min(pbi.diff_turn_points(data_spec))) #30
                                FV = np.append(FV, np.mean(data_spec)) #31
                                FV = np.append(FV, np.std(data_spec)) #32
                                FV = np.append(FV, sc.stats.skew(data_spec)) #33
                                FV = np.append(FV, sc.stats.kurtosis(data_spec, fisher=False)) #34
                                FV = np.append(FV, pbi.shannon_ent(data_spec, 50)) #35
                                FV = np.append(FV, pbi.renyi_ent(data_spec,2, 50)) #36
                                FV = np.append(FV, pbi.RMM(data_spec)) #37
                                FV = np.append(FV, pbi.TCR_f(data_spec, 0.4)) #38
                                FV = np.append(FV, pbi.mTCR_t(data_spec, 0.4)) #39
                                #FV = np.append(FV, np.sum(data_spec)) #40, correlated with mean
                                #FV = np.append(FV, pbi.Erms(data_spec)) #41

                                #New features log of Periods
                                #FV = np.append(FV, np.log10(1/pbi.argmaxEf(data_spec, freq))) #42
                                #FV = np.append(FV, np.log10(1/pbi.centroid_f(data_spec ,freq))) #43
                                #FV = np.append(FV, np.log10(1/pbi.BW_f(data_spec, freq))) #44


                        #MFCC
                        if domain == 'Ceps':
                                #MFCC to FV
                                for mfcc_0 in mfcc_feat:
                                        FV = np.append(FV, mfcc_0) #42-54
                                #for mfcc_1 in d_mfcc_feat:
                                #       FV = np.append(FV, mfcc_1)
                                #for mfcc_2 in dd_mfcc_feat:
                                #       FV = np.append(FV, mfcc_2)

                #print(FV.shape)
        #st_cp.plot()
        return FV


def pb_getpoint(lat0, lon0, edist, back_azm):
	p = LatLon(lat0, lon0)
	d = p.destination(edist*1000, back_azm)
	lat1 = d.lat
	lon1 = d.lon
	return lat1, lon1







