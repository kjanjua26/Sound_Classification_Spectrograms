import glob 
import os
import wave
import numpy as np 
from tqdm import tqdm 
import pylab
from PIL import Image
import matplotlib.pyplot as plt 
from scipy.fftpack import fft

def graph_spectrogram(wav_file):
    sound_info, frame_rate = get_wav_info(wav_file)
    mean_ = np.mean(sound_info)
    std = np.std(sound_info)
    print("Sound Info Type: ", type(sound_info))
    print("Mean: ", np.mean(sound_info))
    print("STD: ", std)
    print("Frame Rate: ", frame_rate)
    mean_norm_wave = np.divide(sound_info, mean_)
    std_norm_wave = np.divide(mean_norm_wave, std)
    fft_out = fft(std_norm_wave)
    print(type(std_norm_wave))
    pylab.figure(num=None, figsize=(19, 12))
    pylab.subplot(111)
    pylab.specgram(fft_out, Fs=frame_rate)
    pylab.savefig('fft_.png')
    pylab.close()

def get_wav_info(wav_file):
    wav = wave.open(wav_file, 'r')
    frames = wav.readframes(-1)
    sound_info = pylab.fromstring(frames, 'Int16')
    frame_rate = wav.getframerate()
    wav.close()
    return sound_info, frame_rate

graph_spectrogram("benedict.wav")
