import glob 
import os
import wave
from tqdm import tqdm 
import pylab

dataset_ = '/Users/Janjua/Desktop/ESC-50-master/dataset/'

def graph_spectrogram(wav_file, class_, filename):
    sound_info, frame_rate = get_wav_info(wav_file)
    pylab.figure(num=None, figsize=(19, 12))
    pylab.subplot(111)
    pylab.title('spectrogram of %r' % wav_file)
    pylab.specgram(sound_info, Fs=frame_rate)
    pylab.savefig('/Users/Janjua/Desktop/ESC-50-master/dataset/{}/{}.png'.format(class_, filename))

def get_wav_info(wav_file):
    wav = wave.open(wav_file, 'r')
    frames = wav.readframes(-1)
    sound_info = pylab.fromstring(frames, 'Int16')
    frame_rate = wav.getframerate()
    wav.close()
    return sound_info, frame_rate

for i in tqdm(glob.glob(dataset_+'*/*')):
	file_name = i.split('/')[-1]
	class_info = i.split('/')[-2]
	#print(file_name)
	graph_spectrogram(i, class_info, file_name)
