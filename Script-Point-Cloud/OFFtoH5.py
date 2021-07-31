import numpy as np
import pandas as pd
import h5py
import os
#from sklearn import preprocessing

#Pottery Categories
"""
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
"""

#Peruvian Categories
categories = [
'basin',
'bowl',
'figurine',
'jar',
'pitcher',
'plate',
'pot',
'vase']


def create_folders(path):
	for i in range(len(categories)):
		command = " ".join(["mkdir",path+categories[i]])
		os.system(command)
		f_train_command = " ".join(["mkdir",path+categories[i]+"/train"])
		os.system(f_train_command)
		f_test_command = " ".join(["mkdir",path+categories[i]+"/test"])
		os.system(f_test_command)


path = './Peruvian_Dataset/'
#path2 = './Pottery_Dataset_PCD/'
path2 = './Peruvian_Dataset_PCD/'
#create_folders(path2)


def OBJtoPCD(path,categories,DataGroup):
	for cat  in categories:
		DataArray=[]
		#deal with train first
		files = os.listdir(path + cat + '/'+DataGroup+'/')
		files = [x for x in files if x[-4:] == '.obj']
		for file_index,file in enumerate(files):
			fileName = file.split('.')[0]
			command = " ".join(["pcl_mesh_sampling",path + cat + '/'+DataGroup+'/' + file,path2 + cat +'/'+DataGroup+'/' + fileName + ".pcd",'-no_vis_result','-n_samples', '2200','-leaf_size', '0.01'])
			os.system(command)            

#OBJtoPCD(path,categories,'train')
#OBJtoPCD(path,categories,'test')

def normalize(x):
	x_mean = np.mean(x,axis=0)
	x_norm = np.copy(x-x_mean)
	x_max = np.max(np.sqrt(np.sum(np.square(x_norm),axis=1)))
	x_norm = x_norm / x_max
	return x_norm	


def PCDtoH5(path,categories,DataGroup):
    for cat  in categories:
        DataArray=[]    
        #deal with train first
        files = os.listdir(path + cat + '/'+DataGroup+'/')
        files = [x for x in files if x[-4:] == '.pcd']
        for file_index,file in enumerate(files):
            fileName = file.split('.')[0]
            with open(path + cat + '/'+DataGroup+'/' + file, 'r') as f:
				for y in range(9):
					f.readline()
				#get number of points in the model
				line = f.readline().replace('\n','')
				point_count = line.split(' ')[1]
				#number of data less or more than 2048
				pad_count = 2048 - int(point_count)
				data = []
				f.readline()
				#fill ndarray with datapoints
				for index in range(0,int(point_count)):
					line = f.readline().rstrip().split()
					line[0] = float(line[0])
					line[1] = float(line[1])
					line[2] = float(line[2])
					data.append(line)
				data = np.array(data)
				
				data = normalize(data)
				#print(data)

				if pad_count > 0 :
					idx = np.random.randint(point_count, size=pad_count)
					data = np.append(data,data[idx],axis=0)
				elif  pad_count < 0 :
					index_pool = np.arange(int(point_count))
					np.random.shuffle(index_pool)
					data = data[index_pool[:2048]]

				data = np.array([data])

				label = np.array(categories.index(cat)).reshape(1,1)
				if file_index == 0 and categories.index(cat) ==0:
					with h5py.File(path + DataGroup +"_Relabel.h5", "w") as ff:
						ff.create_dataset(name='data', data=data,maxshape=(None, 2048, 3), chunks=True)
						ff.create_dataset(name='label', data=label,maxshape=(None, 1), chunks=True)
				else:
					with h5py.File(path +DataGroup +"_Relabel.h5", "a") as hf:
						hf['data'].resize((hf['data'].shape[0] + 1), axis=0)
						hf['data'][-1:] = data
						hf['label'].resize((hf['label'].shape[0] + 1), axis=0)
						hf['label'][-1:] = label


PCDtoH5(path2,categories,'test')
PCDtoH5(path2,categories,'train')

def ShuffleDataSet(path,DataGroup):
    with h5py.File(path +DataGroup+"_Relabel.h5", 'a') as hf:
        label = np.array(hf['label'])
        data = np.array(hf['data'])
        indices = np.arange(data.shape[0])
        np.random.shuffle(indices)

        label = label[indices]
        data = data[indices]
    
        with h5py.File(path + DataGroup +"Shuffled_Relabel.h5", "w") as ff:
            ff.create_dataset(name='data', data=data,shape=(data.shape[0], 2048, 3), chunks=True)
            ff.create_dataset(name='label', data=label,shape=(data.shape[0], 1), chunks=True)