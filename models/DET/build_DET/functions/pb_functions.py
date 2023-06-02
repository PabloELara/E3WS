#E3WS attributes
#Some of these features are in:
#https://doi.org/10.1109/JSTARS.2020.2982714, 
#https://doi.org/10.1109/MSP.2017.2779166

import numpy as np
#from scipy.stats import threshold

#Threshold function from scipy 0.15.1
def threshold(a, threshmin=None, threshmax=None, newval=0):
	a = np.asarray(a).copy()
	mask = np.zeros(a.shape, dtype=bool)
	if threshmin is not None:
		mask |= (a < threshmin)
	if threshmax is not None:
		mask |= (a > threshmax)
	a[mask] = newval
	return a

#Feat 1, length of temporal signal
def lenXt(Xt):
	T = len(Xt)
	return T

#Energy of signal
def Pt(Xt):
	Et = Xt**2
	return Et

#Feat 2, Max of temporal energy
def maxEt(Xt):
	Et = Pt(Xt)
	M = np.max(Et)
	return M

#Erms
def Erms(data):
	E = np.sum(data)
	E_rms = np.sqrt((1/len(data)*E))
	return E_rms

#Feat 3, Mean of temporal energy
#def meanEt(Xt):
#	Et = Pt(Xt)
#	MeanE = np.mean(Et)
#	return MeanE

#Feat 4, Index of max of temporal energy
def argmaxEt(Xt, t):
	Et = Pt(Xt)
	Lt = t[np.argmax(Et)]
	return Lt

#Feat 5, Index of max of PSD
def argmaxEf(PSD, f):
	Lf = f[np.argmax(PSD)]
	return Lf

#Feat 6, Centroid of frequency
def centroid_f(PSD, f):
	Ef = np.sum(PSD)
	centroid = np.sum(f*PSD)/Ef
	return centroid

#Feat 7, Spectrum Spread or Bandwidth (measure of variance around centroid)
def BW_f(PSD, f):
	Ef = np.sum(PSD)
	centroid = centroid_f(PSD,f)
	BWf = np.sqrt( np.sum( (f-centroid)**2 * PSD ) / Ef )
	#BWf = np.sqrt( np.sum((f**2) * (PSD) / Ef) - centroid**2 )
	return BWf

#Feat 8, Frequency skewness, Measure of skewness around bandwidth (This could be negative)
def skewness_f(PSD, f):
	Ef = np.sum(PSD)
	centroid = centroid_f(PSD,f)
	BW = BW_f(PSD,f)
	skewness = np.sum( (f-centroid)**3  * PSD ) / (Ef*BW**3)
	if skewness >= 0:
		return np.sqrt(skewness)
	else:
		return -np.sqrt(-skewness)

#Feat 9, Frequency kurtosis, Measure of kurtosis around BW
def kurtosis_f(PSD, f):
	Ef = np.sum(PSD)
	centroid = centroid_f(PSD,f)
	BW = BW_f(PSD,f)
	kurtosis = np.sqrt(np.sum( (f-centroid)**4  * PSD ) / (Ef*BW**4))
	return kurtosis

#Feat 10, Centroid of time weighted by power
def centroid_t(Xt, t):
	Ene_t = Pt(Xt)
	Et = np.sum(Ene_t)
	centroid = np.sum(t*Ene_t)/Et
	return centroid

#Feat 11, Temporal bandwidth
def BW_t(Xt, t):
	Ene_t = Pt(Xt)
	Et = np.sum(Ene_t)
	centroid = centroid_t(Xt,t)
	BWt = np.sqrt( np.sum( (t-centroid)**2 * Ene_t ) / Et )
	return BWt

#Feat 12, Temporal skewness
def skewness_t(Xt, t):
	Ene_t = Pt(Xt)
	Et = np.sum(Ene_t)
	centroid = centroid_t(Xt,t)
	BW = BW_t(Xt,t)
	skewness = np.sum( (t-centroid)**3  * Ene_t ) / (Et*BW**3)
	if skewness >= 0:
		return np.sqrt(skewness)
	else:
		return -np.sqrt(-skewness)

#Feat 13, Temporal kurtosis
def kurtosis_t(Xt, t):
	Ene_t = Pt(Xt)
	Et = np.sum(Ene_t)
	centroid = centroid_t(Xt,t)
	BW = BW_t(Xt,t)
	kurtosis = np.sqrt(np.sum( (t-centroid)**4  * Ene_t ) / (Et*BW**4))
	return kurtosis

#16, Rate of decay in time
def ROD_t(Xt):
	Ene_t = Pt(Xt)
	delta = Ene_t[1:]-Ene_t[:-1]
	M = np.max(Ene_t)
	rod_t = np.min(delta/M)
	return rod_t

#17, Rate of decay in frequency
def ROD_f(PSD):
	delta = PSD[1:]-PSD[:-1]
	M = np.max(PSD)
	rod_f = np.min(delta/M)
	return rod_f

#18, Ratio of Maximum amplitude envelope to the mean envelope
def RMM(Xn):
	rmm = np.max(Xn)/np.mean(Xn)
	return rmm

#19, scipy skewness envelope
#20, scipy kurtosis envelope

#21, Duration increase respect duration decrease envelope
def IncDec_env(Xenv, env):
	tmax = env[np.argmax(Xenv)]
	ti = env[0]
	tf = env[-1]
	if tf - tmax != 0:
		inc_dec = (tmax - ti)/(tf - tmax)
	else:
		inc_dec = 0
	return inc_dec

#22, Duration increase respect total duration envelope
def Growth_env(Xenv, env):
	tmax = env[np.argmax(Xenv)]
	ti = env[0]
	tf = env[-1]
	if tf - ti != 0:
		growth = (tmax - ti)/(tf - ti)
	else:
		growth = 0
	return growth

#23, How many times the signal exceeds 0, Zero Crossing rate (for example 10.5 times/second)
def ZCR_t (Xn, Fs):
	duration = len(Xn)/Fs
	zcr_t = (((Xn[:-1] * Xn[1:]) < 0).sum())*1/duration
	return zcr_t

#24, Standar deviation envelope

#Ratio of how many times the signal exceeds threshold, for example (0.7 times/second)
def TCR_t (Xn, thres, Fs):
	duration = len(Xn)/Fs
	Xn = Xn/np.max(np.abs(Xn))
	Xn = Xn - thres
	tcr_t = (((Xn[:-1] * Xn[1:]) < 0).sum())*1/duration
	return tcr_t

#Ratio of how many points not exceeds the threshold
def mTCR_t (Xn, thres):
	Xn = Xn/np.max(np.abs(Xn))
	TH = threshold(Xn, threshmin=thres, threshmax=1, newval=-127)
	mtcr_t = np.where(TH != -127)[0] #different of -127 are the elements between thresmin and thresmax
	return len(mtcr_t)/len(Xn)


def shannon_ent (Xn, Bins):
	prob, bins = np.histogram(Xn, bins=Bins)
	prob = prob/len(Xn)
	prob = prob[np.nonzero(prob)]
	shannon_entropy = np.sum(-prob*np.log2(prob))
	return shannon_entropy

def renyi_ent (Xn, alpha, Bins):
	prob, bins = np.histogram(Xn, bins=Bins)
	prob = prob/len(Xn)
	prob = prob[np.nonzero(prob)]
	renyi_entropy = np.log2(np.sum(prob**alpha))/(1-alpha)
	return renyi_entropy


#How many times the signal exceeds threshold, for example (5times in 1 to 20Hz)
def TCR_f (PSD, thres):
	PSD = PSD/np.max(np.abs(PSD))
	PSD = PSD - thres
	tcr_f = (((PSD[:-1] * PSD[1:]) < 0).sum())
	return tcr_f

#Turning points
def group_in_threes(slicable):
	for i in range(len(slicable)-2):
		yield slicable[i:i+3]
def turns(L):
	for index, three in enumerate(group_in_threes(L)):
		if (three[0] > three[1] < three[2]) or (three[0] < three[1] > three[2]):
			yield index + 1
def diff_turn_points(data):
	turn_points_index = list(turns(data))
	E_turn_points = (data)[turn_points_index]
	diff_E_turn_points = np.diff(E_turn_points)
	if len(diff_E_turn_points) == 0: #exception when IMF is generate by sine signal -> PSD has not turning points
		diff_E_turn_points = 0
	return diff_E_turn_points












