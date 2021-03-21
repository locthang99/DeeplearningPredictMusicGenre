import numpy as np
import math
import librosa
from python_speech_features import mfcc
import os,sys

SAMPLE_RATE = 22050
TRACK_DURATION = 30  # measured in seconds
SAMPLES_PER_TRACK = SAMPLE_RATE * TRACK_DURATION

def toMFCC(file_path, num_mfcc, n_fft=2048, hop_length=512, num_segments=10):
    list_mfcc = []
    samples_per_segment = int(SAMPLES_PER_TRACK / num_segments)
    num_mfcc_vectors_per_segment = math.ceil(samples_per_segment / hop_length)

    # load audio file
    signal, sample_rate = librosa.load(file_path, sr=SAMPLE_RATE)
    duration = librosa.get_duration(signal,sample_rate)
    part = int((duration - 15)/30)
    print("Duration: "+str(duration))
    # process all segments of audio file
    i = 0
    while i < part:
        mfcc_part =[]
        for d in range(num_segments):
            start = samples_per_segment * (d + i*10)
            finish = start + samples_per_segment
            mfcc = librosa.feature.mfcc(
                signal[start:finish], sample_rate, n_mfcc=num_mfcc, n_fft=n_fft, hop_length=hop_length)
            mfcc = mfcc.T
            if len(mfcc) == num_mfcc_vectors_per_segment:
                mfcc_part.append(mfcc.tolist())
        # print(np.array(data["mfcc"]))
        x = np.array(mfcc_part)
        x = x[..., np.newaxis]
        list_mfcc.append(x)
        print("save mfcc part "+str(i))
        i+=1
    return list_mfcc

def toMFCC_100(y, sr=16000, nfilt=10, winsteps=0.02):
    try:
        feat = mfcc(y, sr, nfilt=nfilt, winstep=winsteps)
        return feat
    except:
        print("Extraction feature error")

def crop_MFCC_100(feat, i = 0, nb_step=10, maxlen=100):
    sys.stdout = open(os.devnull, 'w')
    crop_feat = np.array(feat[i : i + nb_step]).flatten()
    print(crop_feat.shape)
    crop_feat = np.pad(crop_feat, (0, maxlen - len(crop_feat)), mode='constant')
    sys.stdout = sys.__stdout__
    return crop_feat