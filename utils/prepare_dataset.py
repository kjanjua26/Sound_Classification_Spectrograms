import glob 
import os 

voice_folder = '/Users/Janjua/Desktop/ESC-50-master/audio/'
meta_file = '/Users/Janjua/Desktop/ESC-50-master/meta/esc50.csv'

with open(meta_file, 'r') as mfile:
	lines = mfile.readlines()
	for line in lines:
		filename,_,_,category,_,_,_ = line.split(',')
		for i in glob.glob(voice_folder+'*.wav'):
			file_name = i.split('/')[-1]
			if file_name == filename:
				print('I: ', file_name, 'Class: ', category)
				directory_ = 'dataset/{}'.format(category)
				if(os.path.exists(directory_)):
					os.system('cp {} {}'.format(i, directory_))
				else:
					os.system('mkdir {}'.format(directory_))	
					os.system('cp {} {}'.format(i, directory_))
