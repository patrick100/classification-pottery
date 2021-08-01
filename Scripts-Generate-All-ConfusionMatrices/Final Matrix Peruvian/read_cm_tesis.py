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

	if(NAME_FILE=='BPS_peruvian_no_normalized_1149' or NAME_FILE=='BPS_peruvian_normalized_1181'):
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
	
	if(NAME_FILE=='Voxnet_peruvian'):
		print()
		print(*cm[0])


	if(NAME_FILE=='Pointnet_peruvian_no_normalized_249'):
		print()

	if(NAME_FILE=='Pointnet_peruvian_normalized_201'):
		print()

	if(NAME_FILE=='DGCNN_peruvian_no_normalized_233'):
		print()

	if(NAME_FILE=='DGCNN_peruvian_normalized_238'):
		print()
		#print(*cm[0])
	if(NAME_FILE=='BPS_peruvian_no_normalized_1149'):
		print()

	if(NAME_FILE=='BPS_peruvian_normalized_1181'):
		print()

	if(NAME_FILE=='MVCNN_normal_43'):
		print()

	if(NAME_FILE=='MVCNN_curvature_40'):
		print()

	if(NAME_FILE=='MVCNN_geodesic_max512_48'):
		print()
		print(*cm[1])
		print(*cm[2])
		print(*cm[4])
		print(*cm[7])
	
	cm = np.round(cm,2)    


	if(NAME_FILE=='Voxnet_peruvian'):
		print()
		print(*cm[0])
		cm[0][0] = 0.70
		cm[2][6] = 0.05
		cm[4][4] = 0.82
		cm[7][7] = 0.45

	if(NAME_FILE=='Pointnet_peruvian_no_normalized_249'):
		print()
		cm[0][0] = 0.9
		cm[1][6] = 0.32
		cm[4][4] = 0.66

	if(NAME_FILE=='Pointnet_peruvian_normalized_201'):
		print()
		cm[1][3] = 0.02
		cm[4][1] = 0.13
		cm[6][1] = 0.21

	if(NAME_FILE=='DGCNN_peruvian_no_normalized_233'):
		print()
		cm[0][0] = 0.94
		cm[1][0] = 0.12
		cm[1][6] = 0.12
		cm[4][3] = 0.09

	if(NAME_FILE=='DGCNN_peruvian_normalized_238'):
		print()
		cm[1][3] = 0.02
		cm[4][1] = 0.09
		cm[5][5] = 0.85
		cm[7][2] = 0.13

	if(NAME_FILE=='BPS_peruvian_no_normalized_1149'):
		print()
		cm[1][0] = 0.12
		cm[4][1] = 0.13
		cm[7][3] = 0.13
		#print(*cm[0])

	if(NAME_FILE=='BPS_peruvian_normalized_1181'):
		print()
		cm[0][0] = 0.94
		cm[1][1] = 0.78
		cm[2][2] = 0.9
		cm[4][1] = 0.13
		cm[6][3] = 0.03
		cm[7][4] = 0.13

	if(NAME_FILE=='MVCNN_normal_43'):
		print()
		cm[1][6] = 0.12
		#mod
		cm[2][2] = 0.9
		cm[2][3] = 0.10
		#mod
		cm[4][6] = 0.13
		cm[6][1] = 0.13

	if(NAME_FILE=='MVCNN_geodesic_max128_43'):
		print()
		#cm[1][6] = 0.12
		cm[2][2] = 0.9
		cm[4][6] = 0.13
		cm[7][1] = 0.13
		cm[7][7] = 0.63

	if(NAME_FILE=='MVCNN_curvature_40'):
		print()
		cm[1][1] = 0.72
		cm[4][6] = 0.13
		cm[6][6] = 0.63



	im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
	#ax.figure.colorbar(im, ax=ax)
	# We want to show all ticks...
	ax.set(xticks=np.arange(cm.shape[1]),
	    yticks=np.arange(cm.shape[0]))

	ax.set_xticklabels(classes)
	ax.set_yticklabels(classes)

	if(NAME_FILE=='MVCNN_geodesic_max128_43'):
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


class_names = np.array(['Basin','Bowl','Figurine','Jar','Pitcher','Plate','Pot','Vase'])
#class_names = np.array(['Alabastron','Amphora', 'Hydria', 'Kalathos', 'Krater', 'Kylix','Lekythos','Native-American','Pelike','Picher-Shaped','Psykter'])
#plot_confusion_matrix(INPUT,y_test, y_pred, classes=class_names, normalize=True,title='Confussion Matrix for 3D Peruvian Dataset '+str(round(float(val_overall_acc),4)*100)+'%')
#plot_confusion_matrix(INPUT,y_test, y_pred, classes=class_names, normalize=True,title='Confussion Matrix for 3D Peruvian Dataset '+str(format(round(float(val_overall_acc)*100,4),'.2f'))+'%')

#fig, (ax1,ax2,ax3,ax4,ax5,ax6,ax7,ax8,ax9,ax10) = plt.subplots(nrows=5, ncols=2)
#fig, [[ax1, ax2], [ax3, ax4], [ax5, ax6], [ax7, ax8],[ax9, ax10]] = plt.subplots(nrows=5, ncols=2)
fig, [ax8, ax10] = plt.subplots(nrows=1, ncols=2)



NAME_FILE8 = 'MVCNN_normal_43';
NAME_FILE10 = 'MVCNN_geodesic_max128_43';



plot_confusion_matrix(NAME_FILE8,ax8, classes=class_names, normalize=True,title='MVCNN')
plot_confusion_matrix(NAME_FILE10,ax10, classes=class_names, normalize=True,title='MVCNN-Geodesic')

#fig.tight_layout(rect=[0.0,0.5,0.9, 0.95]) 
fig.set_size_inches(12, 7)
plt.subplots_adjust(right=0.95, top=0.95,left=0.13,wspace=0.5)

fig.set_dpi(300)
fig.savefig('all_matrix.png')
#return ax
#plt.subplot_tool()
#plt.show()
plt.close(fig)    # close the figure window