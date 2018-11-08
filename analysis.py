import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import numpy as np
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc, confusion_matrix
from scipy import interp
import itertools




def plot_confusion_matrix(label_list, pred_list, classes,dir_out,
                          normalize=True,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and saves the plot of the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """

    file_name = dir_out+"confusion_matrix"
    for c in classes:
        file_name= file_name+"_"+c
    file_name = file_name.replace(" ","_")
    file_name=file_name+".pdf"
    pdf = PdfPages(file_name)
    fig = plt.figure()

    cm = confusion_matrix(label_list, pred_list)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    pdf.savefig(fig)
    plt.close(fig)
    pdf.close()

def plot_epochs(y,y_label,dir_out,x=None):
    """
    This function prints and saves the plot of some value that changes each epoch
    such as the training and validtion loss
    """

    fname = dir_out+y_label
    fname = fname.replace(" ","_")+"_over_epochs.pdf"
    pdf = PdfPages(fname)
    fig = plt.figure()

    if x==None:
        x=np.arange(len(y))

    plt.plot(x, y)
    plt.xlabel("epoch")
    plt.xticks(x)
    plt.ylabel(y_label)
    plt.title(y_label+" over epochs")

    pdf.savefig(fig)
    plt.close(fig)
    pdf.close()

def plot_roc(label_list,prob_list,classes,dir_out):
    fname = dir_out+"roc_curve"
    for c in classes:
        fname = fname+"_"+c
    fname = fname.replace(" ","_")+".pdf"
    pdf = PdfPages(fname)
    fig = plt.figure()

    class_numbs = len(classes)
    lw = 2
    colors = ['b','g','r','c','m','y','k']


    if class_numbs==2:
        fpr, tpr, thresholds = roc_curve(np.asarray(label_list), np.asarray(prob_list)[:,1])
        roc_auc = auc(fpr,tpr)
        plt.plot(fpr, tpr, color='blue',
                 lw=lw, label=classes[0]+'ROC curve (area = %0.2f)' % roc_auc)


    if class_numbs>2:
        n_classes = np.arange(class_numbs)
        y_roc = label_binarize(np.asarray(label_list), classes=n_classes)
        fpr     = dict()
        tpr     = dict()
        roc_auc = dict()
        for i in n_classes:
            fpr[i],tpr[i],_ = roc_curve(y_roc[:,i],np.asarray(prob_list)[:,i])
            roc_auc[i]      = auc(fpr[i],tpr[i])
            plt.plot(fpr[i], tpr[i], color=colors[i],
                 lw=lw, label=classes[i]+' ROC curve (area = %0.2f)' % roc_auc[i])


    plt.plot([0, 1], [0, 1], color='red', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")



    pdf.savefig(fig)
    plt.close(fig)
    pdf.close()
