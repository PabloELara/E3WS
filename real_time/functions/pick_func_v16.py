import numpy as np
import functions.pb_utils_v16 as pb_utils

def PP_pick(st_pb, mov_time, pb_inst, pzfile, PICK_pb, fmin, fmax):
	Fs = st_pb[0].stats.sampling_rate
	slidding_seconds = mov_time
	N = len(st_pb[0].data)
	w_sec = 4
	n_waves = int((N/Fs-w_sec)/slidding_seconds)+1

	PROB_PP = np.array([])
	for i4 in range(0, n_waves):
		st_pick = st_pb.copy()

		t3 = i4*slidding_seconds
		t4 = w_sec+t3

		st_pick = st_pick.trim(st_pick[0].stats.starttime+t3, st_pick[0].stats.starttime+t4)

		if pb_inst == True:
			pzfile = pzfile
		else:
			pzfile = ''

		#Feature vector
		FV = pb_utils.st_FV(st_pick, pb_inst, pzfile=pzfile, fmin=fmin, fmax=fmax)
		FV = np.real([FV]) #Just for 3C the same in all channels

		p_pickp = PICK_pb.predict_proba(FV)[0][1] #classification

		#print(i4, n_waves, st_pick[0].stats.starttime, st_pick[0].stats.endtime, p_pickp)
		PROB_PP = np.append(PROB_PP, p_pickp)

	return PROB_PP
