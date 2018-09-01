from bs4 import BeautifulSoup
import requests
import re
import urllib2
import os
import cookielib
import json
import glob 

data_set = 'ESC-50-master/dataset/'

def get_soup(url,header):
    return BeautifulSoup(urllib2.urlopen(urllib2.Request(url,headers=header)),'html.parser')


def get_images(query, class_name):
	print('For Query: ', query)
	image_type="ActiOn"
	url = "https://www.google.co.in/search?q="+query+"&source=lnms&tbm=isch"
	header = {'User-Agent':"Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/43.0.2357.134 Safari/537.36"}
	soup = get_soup(url,header)
	ActualImages = [] # contains the link for Large original images, type of  image
	for a in soup.find_all("div",{"class":"rg_meta"}):
	    link , Type =json.loads(a.text)["ou"],json.loads(a.text)["ity"]
	    ActualImages.append((link,Type))

	for i in range(len(ActualImages)):
		file_name, type_ = ActualImages[i]
		print(i,file_name)
		os.system('wget -O ESC-50-master/dataset/{}/{}.png {}'.format(class_name, i, file_name))


for i in glob.glob(data_set+'*'):
	class_name = i.split('/')[-1]
	get_images(class_name,class_name)