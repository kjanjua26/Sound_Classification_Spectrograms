import glob 
import os
import wave
from tqdm import tqdm 
import pylab
import numpy as np
from scipy import signal
from scipy.io import wavfile
import matplotlib.pyplot as plt

dataset_ = 'dataset/'
rem_list = ['jackhammer', 'gun_shot', 'drilling', 'air_conditioner', 'siren', 'car_horn', 'children_playing', 'dog_bark', 'engine_idling', 'street_music']
def graph_spectrogram(wav_file, class_, filename):
    sound_info, frame_rate = get_wav_info(wav_file)
    pylab.figure(num=None, figsize=(19, 12))
    pylab.subplot(111)
    pylab.specgram(sound_info, Fs=frame_rate)
    pylab.savefig('{}.png'.format(filename))
    pylab.close()

def get_wav_info(wav_file):
    wav = wave.open(wav_file, 'r')
    frames = wav.readframes(-1)
    sound_info = pylab.fromstring(frames, 'Int16')
    frame_rate = wav.getframerate()
    wav.close()
    return sound_info, frame_rate

def convert_to_2channel(wav_file):
    temp = wav_file.split('.')[0].replace("twochanel","")
    out_name = str(temp) + ".wav"
    os.popen("ffmpeg -i {} -ac 2 {}".format(wav_file, out_name), "w")

def clear_two_chanel(wav_file):
    if 'twochanel' in wav_file:
        os.system("rm {}".format(wav_file))

for j in rem_list:
    for i in tqdm(glob.glob(dataset_+'{}/*.wav'.format(j))):
        convert_to_2channel(i)
    for i in tqdm(glob.glob(dataset_+'{}/*.wav'.format(j))):
        clear_two_chanel(i)
    for i in tqdm(glob.glob(dataset_+'{}/*.wav'.format(j))):
        file_name = i.split('/')[-1]
        class_info = i.split('/')[-2]
        try:
            graph_spectrogram(i, class_info, file_name)
        except:
            print("Failed: ", i)
