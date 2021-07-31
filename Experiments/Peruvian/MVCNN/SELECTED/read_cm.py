import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels

import math  

def plot_confusion_matrix(INPUT,y_true, y_pred, classes,
                        normalize=False,
                        title=None,
                        cmap=plt.cm.Greys):

    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
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
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    cm = np.round(cm,2)    
    """
    print(*cm[7])
    cm = np.round(cm,2)

    if(INPUT=='249.npz'):
        cm[1][4] = 0.02
        cm[2][2] = 0.9
        cm[4][6] = 0.13
        cm[7][2] = 0.13

    print(*cm[7])
	"""

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    #ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        # ... and label them with the respective list entries
        xticklabels=classes, yticklabels=classes,
        title=title,
        ylabel='True label',
        xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
            rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    #print(cm)
    for i in range(cm.shape[0]):
        #print(cm[i], end=" ")
        #print(*cm[i])

        temp=0 
        for j in range(cm.shape[1]):
            if(cm[i, j] == 0):
                ax.text(j, i, format(cm[i, j], '.0f'),ha="center", va="center",color="white" if cm[i, j] > thresh else "black")
            else:
                ax.text(j, i, cm[i, j],ha="center", va="center",color="white" if cm[i, j] > thresh else "black")
            
            #if(cm[i, j] == 0):
            #    ax.text(j, i, format(cm[i, j], '.0f'),ha="center", va="center",color="white" if cm[i, j] > thresh else "black")
            #else:
                #if(INPUT=='max512.npz'):
                #    ax.text(j, i, format(cm[i, j], fmt),ha="center", va="center",color="white" if cm[i, j] > thresh else "black")
            temp+= round(cm[i, j],2)  
        if(temp!=1):
            print("SUM LINE "+str(i+1)+" :",temp)    
    fig.tight_layout()
    fig.savefig('confusion_matrix_pottery_'+INPUT+'.png')

    #return ax
    plt.show()
    plt.close(fig)    # close the figure window

#np.savez('data_matrix_confusion_max512.npz', y_test=y_test, y_pred=y_pred)

INPUT = '43.npz'

data = np.load('data_matrix_confusion_normal_'+INPUT)
y_test = data['y_test']
y_pred = data['y_pred']
val_overall_acc = data['val_overall_acc']
#print("Y_TEST")
#print(y_test)
#print("Y_PRED")
#print(y_pred)
class_names = np.array(['Basin','Bowl','Figurine','Jar','Pitcher','Plate','Pot','Vase'])
#class_names = np.array(['Alabastron','Amphora', 'Hydria', 'Kalathos', 'Krater', 'Kylix','Lekythos','Native-American','Pelike','Picher-Shaped','Psykter'])
#plot_confusion_matrix(INPUT,y_test, y_pred, classes=class_names, normalize=True,title='Confussion Matrix for 3D Peruvian Dataset '+str(round(float(val_overall_acc),4)*100)+'%')
plot_confusion_matrix(INPUT,y_test, y_pred, classes=class_names, normalize=True,title='Confussion Matrix for 3D Peruvian Dataset '+str(format(round(float(val_overall_acc)*100,4),'.2f'))+'%')
