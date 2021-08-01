import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels

import math  

def plot_confusion_matrix(NAME_FILE,ax, classes,
                        normalize=False,
                        title=None,
                        cmap=plt.cm.Blues):

	"""
	This function prints and plots the confusion matrix.
	Normalization can be applied by setting `normalize=True`.
	"""

	print(NAME_FILE)
	data = np.load(NAME_FILE+'.npz')
	y_true = data['y_test']
	y_pred = data['y_pred']
	val_overall_acc = data['val_overall_acc']

	if(NAME_FILE=='BPS_pottery_no_normalized_1110' or NAME_FILE=='BPS_pottery_normalized_1179'):
		title = title+' '+str(format(val_overall_acc,'.2f'))+'%' 
	else:
		title = title+' '+str(format(val_overall_acc*100,'.2f'))+'%' 
		#str(round(float(val_overall_acc),4)*100)+'%')

	if not title:
	    if normalize:
	        title = 'Normalized confusion matrix'
	    else:
	        title = 'Confusion matrix, without normalization'

	# Compute confusion matrix
	cm = confusion_matrix(y_true, y_pred)
	#cm = np.around(cm,3)

	# Only use the labels that appear in the data
	classes = classes[unique_labels(y_true, y_pred)]
	if normalize:
	    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
	    #print("Normalized confusion matrix")
	#else:
	    #print('Confusion matrix, without normalization')
	
	if(NAME_FILE=='Pointnet_pottery_no_normalized_247'):
		print()
		#print(*cm[6])
		#print(*cm[9])
	if(NAME_FILE=='Pointnet_pottery_normalized_247'):
		print()

	if(NAME_FILE=='DGCNN_pottery_no_normalized_243'):
		print()

	if(NAME_FILE=='DGCNN_pottery_normalized_204'):
		print()

	cm = np.round(cm,2)    

	if(NAME_FILE=='Pointnet_pottery_no_normalized_247'):
		#print()
		#print(*cm[2])
		cm[1][2] = 0.04
		cm[6][6] = 0.66
		cm[9][2] = 0.13

	if(NAME_FILE=='Pointnet_pottery_normalized_247'):
		cm[6][6] = 0.66
		cm[9][2] = 0.13

	if(NAME_FILE=='DGCNN_pottery_no_normalized_243'):
		cm[6][6] = 0.66

	if(NAME_FILE=='DGCNN_pottery_normalized_204'):
		cm[1][6] = 0.04


	im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
	#ax.figure.colorbar(im, ax=ax)
	# We want to show all ticks...
	ax.set(xticks=np.arange(cm.shape[1]),
	    yticks=np.arange(cm.shape[0]))

	ax.set_xticklabels(classes)
	ax.set_yticklabels(classes)

	if(NAME_FILE=='MVCNN_geodesic_max1_44'):
		ax.set_title(title, style='italic', fontweight='bold')
	else:
		ax.set_title(title, style='italic')


	ax.set_xlabel('Predicción')
	ax.set_ylabel('Verdadera')

	# Rotate the tick labels and set their alignment.
	plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
	        rotation_mode="anchor")

	# Loop over data dimensions and create text annotations.
	fmt = '.2f' if normalize else 'd'
	thresh = cm.max() / 2.
	for i in range(cm.shape[0]):
	    temp=0 
	    for j in range(cm.shape[1]):
	        if(cm[i, j] == 0):
	            ax.text(j, i, format(cm[i, j], '.0f'),ha="center", va="center",color="white" if cm[i, j] > thresh else "black")
	        else:
	            ax.text(j, i, cm[i, j],ha="center", va="center",color="white" if cm[i, j] > thresh else "black")
	      
	        temp+=cm[i, j] 
	    if(temp!=1):
	        print("SUM LINE "+str(i+1)+" :",temp)
	print()    


#class_names = np.array(['Basin','Bowl','Figurine','Jar','Pitcher','Plate','Pot','Vase'])
class_names = np.array(['Alabastron','Amphora', 'Hydria', 'Kalathos', 'Krater', 'Kylix','Lekythos','Native-American','Pelike','Picher-Shaped','Psykter'])
#plot_confusion_matrix(INPUT,y_test, y_pred, classes=class_names, normalize=True,title='Confussion Matrix for 3D Peruvian Dataset '+str(round(float(val_overall_acc),4)*100)+'%')
#plot_confusion_matrix(INPUT,y_test, y_pred, classes=class_names, normalize=True,title='Confussion Matrix for 3D Peruvian Dataset '+str(format(round(float(val_overall_acc)*100,4),'.2f'))+'%')

#fig, (ax1,ax2,ax3,ax4,ax5,ax6,ax7,ax8,ax9,ax10) = plt.subplots(nrows=5, ncols=2)
fig, [ax8, ax9] = plt.subplots(nrows=1, ncols=2)


NAME_FILE8 = 'MVCNN_normal_50';
NAME_FILE9 = 'MVCNN_geodesic_max1_44';


plot_confusion_matrix(NAME_FILE8,ax8, classes=class_names, normalize=True,title='MVCNN')
plot_confusion_matrix(NAME_FILE9,ax9, classes=class_names, normalize=True,title='MVCNN-Geodesic')


#fig.tight_layout(rect=[0.0,0.5,0.9, 0.95]) 
fig.set_size_inches(12, 7)
plt.subplots_adjust(right=0.95, top=0.95,left=0.13,wspace=0.5)
#plt.subplot_tool()

#plt.subplots_adjust(left=0, bottom=0.05, right=1, top=0.95, wspace=0, hspace=0.7)

fig.set_dpi(300)
fig.savefig('all_matrix.png')
#return ax
#plt.subplot_tool()
#plt.show()
plt.close(fig)    # close the figure window