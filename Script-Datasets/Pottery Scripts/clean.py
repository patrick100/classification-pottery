#!/usr/bin/env python
# coding: utf-8

# The whole process begins with raw ModelNet10 data(.OFF file)
# It only contains endpoints of the model(like those points at each corner).
# Inorder to get points that evenly spread across all surfacts of the model, we need PointCloudLibrary(PCL) to sample our model.
# but PCL only accept .PLY file so conversion is needed.

# First: convert .OFF file to .PLY file.

# In[1]:


import numpy as np
import pandas as pd
import h5py
import os
from sklearn import preprocessing
from functools import partial
from multiprocessing.dummy import Pool
from subprocess import call


categories = ['Alabastron',
'Amphora',
'Hydria',
'Kalathos',
'Krater',
'Kylix',
'Lekythos',
'Native-American',
'Pelike',
'Picher-Shaped',
'Psykter']

def create_folders(path):
	for i in range(len(categories)):
		command = " ".join(["mkdir",path+categories[i]])
		os.system(command)
		f_train_command = " ".join(["mkdir",path+categories[i]+"/train"])
		os.system(f_train_command)
		f_test_command = " ".join(["mkdir",path+categories[i]+"/test"])
		os.system(f_test_command)

path = './pottery_dataset/'
path1 = './pottery_dataset_cleaned/'
create_folders(path1)

def convert_txt(path,categories,DataGroup):
	commands = []
	for cat  in categories:
		#deal with train first
		files = os.listdir(path + cat + '/'+DataGroup+'/')
		files = [x for x in files if x[-4:] == '.obj']
		for file_index,file in enumerate(files):
			fileName = file.split('.')[0]
			output_name = fileName.replace(" ","-")
			#for i in range(0, nviews):
			#print(str(stepsize*i))
			#rotation = (offset+(stepsize*i)*-1)
			command = " ".join(["./meshconv","'"+path + cat + '/'+DataGroup+'/' + file+"'","-c","obj","-tri","-o",path1 + cat +'/'+DataGroup+'/' + output_name])
			commands.append(command)			
			print(command)			
			#os.system(command)
			#print(command)
	
	
	pool = Pool(12) # two concurrent commands at a time
	for i, returncode in enumerate(pool.imap(partial(call, shell=True), commands)):
		if(returncode != 0):
			print("%d command failed: %d" % (i, returncode))   
	

convert_txt(path,categories,'train')
convert_txt(path,categories,'test')
