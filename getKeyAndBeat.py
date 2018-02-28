import math
import numpy as np
from scipy.io import loadmat
from scipy.signal import upfirdn, filtfilt
from librosa.core import load, resample
from librosa.beat import beat_track
from matplotlib import pyplot as plt
from matplotlib import mlab
import itertools
import operator
import os
import ipdb
import sys


def audio_to_pitch_via_FB(f_audio,
						winLenSTMSP=4410,
						midiMin=21,
						midiMax=108,
						fs=22050):

	'''
	Computing and saving of pitch features via a pre-designed filterbank.
	features. For each window length specified via winLenSTMSP
	the following is computed:
	- STMSP (short-time mean-square power) for each MIDI pitch between
		midiMin and midiMax
	- STMSP subbands are stored in f_pitch, where f_pitch(p,:) contains
		STMSP of subband of pitch p

	Output:
		f_pitch
	'''
	x = loadmat('MIDI_FB_ellip_pitch_60_96_22050_Q25.mat')
	h = x['h']

	pcm_ds = {}
	pcm_ds[1] = f_audio
	pcm_ds[2] = resample(pcm_ds[1], 5, 1, res_type='kaiser_fast')
	pcm_ds[3] = resample(pcm_ds[2], 5, 1, res_type='kaiser_fast')

	fs_pitch = np.zeros((128,))
	fs_index = np.zeros((128,), dtype=int)
	fs_pitch[20:59] = 882
	fs_pitch[59:95] = 4410
	fs_pitch[95:120] = 22050
	fs_index[20:59] = 3
	fs_index[59:95] = 2
	fs_index[95:120] = 1
	print('Computing subbands and STMSP for all pitches: ' + str( midiMin) + ' - ' + str(midiMax))
	# Compute features for all pitches

	winOvSTMSP = np.round(winLenSTMSP / 2)
	featureRate = float(fs) / (winLenSTMSP - winOvSTMSP)

	wav_size = f_audio.shape[0]

	step_size = winLenSTMSP - winOvSTMSP
	group_delay = np.round(winLenSTMSP / 2)
	seg_pcm_start = np.concatenate(([0], range(0, wav_size, step_size)))
	seg_pcm_stop = np.copy(seg_pcm_start)
	seg_pcm_stop += winLenSTMSP
	seg_pcm_stop[np.where(seg_pcm_stop > wav_size)] = wav_size
	seg_pcm_stop[0] = min(group_delay, wav_size)
	seg_pcm_num = len(seg_pcm_start)
	f_pitch_energy = np.zeros((120, seg_pcm_num))

	for p in range(midiMin, midiMax):
		index = fs_index[p]
		f_filtfilt=filtfilt(h[0,p][1][0], h[0,p][0][0], pcm_ds[index])
		f_square=f_filtfilt ** 2
		# f_pitch_energy
		factor = (fs / fs_pitch[p])
		for k in range(0, seg_pcm_num):
			start = int(np.ceil((float(seg_pcm_start[k]) / fs) * fs_pitch[p]))
			stop = int(np.floor((float(seg_pcm_stop[k]) / fs) * fs_pitch[p]))
			f_pitch_energy[p,k] = sum(f_square[start:stop]) * factor

	f_pitch = f_pitch_energy
	return f_pitch, featureRate, winLenSTMSP, winOvSTMSP


def smoothDownsampleFeature(f_feature,
							winLenSmooth=1,
							downsampSmooth=1,
							inputFeatureRate=0):

	'''
	- Temporal smoothing and downsampling of a feature sequence

	- parameter.featureRate specifies the input feature rate. This value is
	used to derive the output feature rate.


	Output:
			  f_feature
			  newFeatureRate
	'''

	# Temporal Smoothing
	if winLenSmooth != 1 or downsampSmooth != 1:
		stat_window = np.hanning(winLenSmooth+2)
		stat_window /= np.sum(stat_window)
		f_feature_stat = np.zeros(f_feature.shape)
		f_feature_stat = upfirdn(stat_window, f_feature, 1, downsampSmooth)
		seg_num = f_feature.shape[1]
		stat_num = np.ceil(seg_num / downsampSmooth)
		cut = np.floor((winLenSmooth - 1) / (2 * downsampSmooth))
		f_feature_stat = f_feature_stat[:,range(int(cut), int(stat_num + cut))]
	else:
		f_feature_stat = f_feature

	newFeatureRate = float(inputFeatureRate) / downsampSmooth
	return f_feature_stat, newFeatureRate



def normalizeFeature(f_feature=None,normP=None,threshold=None,*args,**kwargs):

	'''
	- Normalizes a feature sequence according to the l^p norm
	- If the norm falls below threshold for a feature vector, then the
	normalized feature vector is set to be the unit vector.

	Output:
	f_featureNorm
	'''

	f_featureNorm = np.zeros(f_feature.shape)
	unit_vec = np.ones((1,12))
	unit_vec /= np.linalg.norm(unit_vec, ord=normP)
	for k in range(0, f_feature.shape[1]):
		n = np.linalg.norm(f_feature[:,k], normP)
		if n < threshold:
			f_featureNorm[:,k] = unit_vec
		else:
			f_featureNorm[:,k] = f_feature[:,k] / n

	return f_featureNorm


def pitch_to_CRP(f_pitch,
			  	coeffsToKeep=range(54,120),
				applyLogCompr=True,
				factorLogCompr=1000,
				addTermLogCompr=1,
				normP=2,
				winLenSmooth=1,
				downsampSmooth=1,
				normThresh=10**-6,
				inputFeatureRate=0):
	'''
	Calculates CRP (Chroma DCT-reduced Log Pitch) Features
	(see "Towards Timbre-Invariant Audio Features for Harmony-Based Music" by
	Meinard Mueller and Sebastian Ewert)
	'''
	seg_num = f_pitch.shape[1]
	# log compression
	if applyLogCompr:
		f_pitch_log = np.log10(addTermLogCompr + f_pitch * factorLogCompr)
	else:
		f_pitch_log = f_pitch

	# DCT based reduction
	DCT = internal_DCT(f_pitch_log.shape[0])
	DCTcut = np.copy(DCT)
	DCTcut[np.setdiff1d(range(0,120), coeffsToKeep), :] = 0
	DCT_filter = np.dot(np.transpose(DCT), DCTcut)
	f_pitch_log_DCT = np.dot(DCT_filter, f_pitch_log)
	# calculate energy for each chroma band
	f_CRP = np.zeros((12, seg_num))
	for p in range(0,120):
		chroma = np.mod(p+1,12)
		f_CRP[chroma,:] = f_CRP[chroma,:] + f_pitch_log_DCT[p,:]
	# normalize the vectors according to the norm l^p
	f_CRP = normalizeFeature(f_CRP,normP,normThresh)
	if (winLenSmooth != 1) or (downsampSmooth != 1):
		# Temporal smoothing and downsampling
		f_CRP, CrpFeatureRate = smoothDownsampleFeature(f_CRP,
														winLenSmooth,
														downsampSmooth,
														inputFeatureRate)
		f_CRP = normalizeFeature(f_CRP,normP,normThresh)
	else:
		CrpFeatureRate = inputFeatureRate

	return f_CRP


def internal_DCT(l=None,*args,**kwargs):
	matrix = np.zeros((l,l))
	for m in range(0, l):
		for n in range(0, l):
			matrix[m, n] = np.sqrt(2/float(l)) * np.cos((m * (n+ 0.5) * np.pi) / float(l))

	matrix[0,:] /= np.sqrt(2)
	return matrix


def gausswin(L,a=2.5):
	'''
	GAUSSWIN(N) returns an N-point Gaussian window.

	GAUSSWIN(N, ALPHA) returns the ALPHA-valued N-point Gaussian
	window.  ALPHA is defined as the reciprocal of the standard
	deviation and is a measure of the width of its Fourier Transform.
	As ALPHA increases, the width of the window will decrease. If omitted,
	ALPHA is 2.5.

	Reference:
	[1] fredric j. harris [sic], On the Use of Windows for Harmonic
	Analysis with the Discrete Fourier Transform, Proceedings of
	the IEEE, Vol. 66, No. 1, January 1978
	'''

	N = L
	n = np.array(range(0, N))-(N-1)/2.
	w = np.exp(-0.5*(a*n/ ((N-1)/2.))**2.)
	return w


def estimateTuning(f_input,
				numAdditionalTunings=0,
				pitchRange=np.array(range(21,109)),
				fftWindowLength=8192,
				windowFunction=np.hanning):
	'''
	- input is a mono audio signal
	- sampling freq of 22050 Hz is assumed
	- guesses the tuning according to a simple energy maximizing criterion
	- output is either: what shiftFB is best to use (shiftFB \in [0:5]).
	Alternatively, the center freq for A4 is given which can be used to
	specify a filterbank on your own. The second option is more fine grained.
	Alternatively, it gives a tuning in semitones, which can easily be shifted
	cyclicly. For example: a tuning of -19/20 is more likely to be
	+1/20 Tuning difference.
	'''


	pitchWeights = gausswin(pitchRange.size) ** 2.
	numTunings = 6 + numAdditionalTunings
	referenceFreqsA4 = np.zeros((numTunings,))
	tunings = np.zeros((numTunings,))
	tunings[0] = 0
	tunings[1] = -1./4
	tunings[2] = -1./3
	tunings[3] = -1./2
	tunings[4] = -2./3
	tunings[5] = -3./4

	for k in range(0, numAdditionalTunings):
		tunings[k+6] = -k / (numAdditionalTunings+1)

	referenceFreqsA4 = 2**((69-69+tunings)/12)*440
	#NFFT = int(max(256, 2**np.ceil(math.log(fftWindowLength, 2))))
	NFFT = fftWindowLength
	window = windowFunction(fftWindowLength)
	s,f,t, im = plt.specgram(f_input, window=window, NFFT=NFFT, noverlap=fftWindowLength/2, mode='magnitude', Fs=22050)
	s = abs(s)
	directFreqBinSearch = 0
	if all((f[1:]-f[0:-1])-(f[1]-f[0]) < np.spacing(1)):
		directFreqBinSearch = 1
	averagedPowerSpectrogram = np.sum(s**2., axis=1)
	totalPitchEnergyViaSpec = np.zeros((numTunings,))

	for tu in range(0, numTunings):
		centerfreqs = 2.**((pitchRange-69.0)/12)*referenceFreqsA4[tu]
		upperborderfreqs = 2.**((pitchRange-68.5)/12)*referenceFreqsA4[tu]
		lowerborderfreqs = 2.**((pitchRange-69.5)/12)*referenceFreqsA4[tu]
		spectrogramFilter = np.zeros((len(f),))
		for k in range(0, len(pitchRange)):
			c = getCorrespondingBin(f,centerfreqs[k],directFreqBinSearch)
			u = getCorrespondingBin(f,upperborderfreqs[k],directFreqBinSearch)
			l = getCorrespondingBin(f,lowerborderfreqs[k],directFreqBinSearch)
			spectrogramFilter[c:u+1] = np.multiply(pitchWeights[k], np.linspace(1, 0, u-c+1))
			spectrogramFilter[l:c+1] = np.multiply(pitchWeights[k], np.linspace(0, 1, c-l+1))
		totalPitchEnergyViaSpec[tu] = sum(np.multiply(spectrogramFilter**2.,
												averagedPowerSpectrogram))
	shiftFB = np.argmax(totalPitchEnergyViaSpec[0:6])
	maxIndex = np.argmax(totalPitchEnergyViaSpec)
	centerA4 = referenceFreqsA4[maxIndex]
	tuningSemitones = tunings[maxIndex]

	return shiftFB, centerA4, tuningSemitones


def getCorrespondingBin(x, sval, directSearch):
	if sval >= x[-1]:
		index = len(x) - 1
		return index
	elif sval <= x[0]:
		index = 0
		return index
	if directSearch:
		index = np.round((sval-x[0]) / (x[1]-x[0]))
	else:
		fr = 0
		to = len(x)-1
		while fr <= to:
			mid = np.round((fr + to)/2)
			diff = x[mid] - sval
			if diff < 0:
				fr = mid
			else:
				to = mid
			if to - fr == 1:
				break
		if np.abs(x[fr]-sval) < np.abs(x[to]-sval):
			index = fr
		else:
			index = to
	return int(index)


def get_notes_and_chords(snippet):
	chroma_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
	indices = np.where(snippet>0.1)[0]
	notes = [chroma_names[i] for i in indices]
	chords = []
	for i in range(0, len(chroma_names)):
		if chroma_names[i] in notes:
			note_major1 = chroma_names[np.mod(i+4, 12)]
			note_2 = chroma_names[np.mod(i+7, 12)]
			note_minor1 = chroma_names[np.mod(i+3, 12)]
			if note_major1 in notes and note_2 in notes:
				chords.append(chroma_names[i] + 'maj')
			if note_minor1 in notes and note_2 in notes:
				chords.append(chroma_names[i] + 'min')
	return notes, chords

def most_common(L):
  # get an iterable of (item, iterable) pairs
  SL = sorted((x, i) for i, x in enumerate(L))
  # print 'SL:', SL
  groups = itertools.groupby(SL, key=operator.itemgetter(0))
  # auxiliary function to get "quality" for an item
  def _auxfun(g):
    item, iterable = g
    count = 0
    min_index = len(L)
    for _, where in iterable:
      count += 1
      min_index = min(min_index, where)
    # print 'item %r, count %r, minind %r' % (item, count, min_index)
    return count, -min_index
  # pick the highest-count/earliest item
  return max(groups, key=_auxfun)[0]


def get_key(f):
	all_chords = []
	for i in range(0, f.shape[1]):
		notes, chords = get_notes_and_chords(f[:,i])
		all_chords += chords
	key = most_common(all_chords)
	all_chords = [c for c in all_chords if c != key]
	if len(all_chords) > 0:
		key2 = most_common(all_chords)
	else:
		key2 = None
	return key, key2


def main(filename):
	f_audio, fs = load(filename)
	print f_audio.size
	if f_audio.size > 500000:
		f_start = f_audio[0:500000]
		f_end = f_audio[-500000:-1]
	else:
		f_start = f_audio
		f_end = f_audio
	both_fs = [f_start, f_end]
	
	for i in range(0,2):
		f_audio = both_fs[i]
		shiftFB=estimateTuning(f_audio)
		print 'shiftFB: ' + str(shiftFB)
		f_pitch, featureRate, winLenSTMSP, winOvSTMSP = audio_to_pitch_via_FB(f_audio)
		print 'finished audio_to_pitch_via_FB'
		f_CRP  = pitch_to_CRP(f_pitch, inputFeatureRate=featureRate)
		f_CRPSmoothed, featureRateSmoothed = smoothDownsampleFeature(f_CRP, 
											winLenSmooth=21,
											downsampSmooth=5,
											inputFeatureRate=featureRate)
		key, key2 = get_key(f_CRPSmoothed)
		tempo, beat = beat_track(f_audio, fs)
		if i == 0:
			start_stats = [key, key2, tempo]
			print 'start_stats: keys = ' + key + ', ' + key2 + ' and tempo = ' + str(tempo)
		elif i == 1:
			end_stats = [key, key2, tempo]
			print 'end_stats: keys = ' + key + ', ' + key2 + ' and tempo = ' + str(tempo)
	return start_stats, end_stats


if __name__ == '__main__':
	filename = sys.argv[1]
	main(filename)
