import glob 
import os
import wave
from tqdm import tqdm 
import pylab
import numpy as np

dataset_ = '/Users/Janjua/Desktop/ItalyWork/Multimodal_DL/Sound_Imag/ESC-50-master/dataset/'

def graph_spectrogram(wav_file, class_, filename):
    sound_info, frame_rate = get_wav_info(wav_file)
    frame_rate = 16000
    mean_ = np.mean(sound_info)
    std = np.std(sound_info)
    print("====================================")
    print("Mean: ", np.mean(sound_info))
    print("STD: ", std)
    print("Frame Rate: ", frame_rate)
    print("====================================")
    mean_norm_wave = np.divide(sound_info, mean_)
    std_norm_wave = np.divide(mean_norm_wave, std)
    pylab.figure(num=None, figsize=(14, 8))
    pylab.subplot(111)
    pylab.title('spectrogram of %r' % wav_file)
    pylab.specgram(std_norm_wave, Fs=frame_rate)
    pylab.savefig('/Users/Janjua/Desktop/ItalyWork/Multimodal_DL/Sound_Imag/ESC-50-master/dataset/{}/{}.png'.format(class_, filename))
    pylab.close()

def get_wav_info(wav_file):
    wav = wave.open(wav_file, 'r')
    frames = wav.readframes(-1)
    sound_info = pylab.fromstring(frames, 'Int16')
    frame_rate = wav.getframerate()
    wav.close()
    return sound_info, frame_rate

for i in tqdm(glob.glob(dataset_+'*/*.wav')):
    file_name = i.split('/')[-1]
    class_info = i.split('/')[-2]
    graph_spectrogram(i, class_info, file_name)
