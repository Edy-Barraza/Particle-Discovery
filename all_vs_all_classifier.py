from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
from torchvision import models, transforms, datasets
import numpy as np
import matplotlib.pyplot as plt
import itertools
import time
import os
import copy
import sys


from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc
from scipy import interp

plt.ion()   # interactive mode




PATH = "../../data/images_600/all_vs_all/"





#characteristics of data load

input_shape = 600
batch_size  = 6

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]



#apply data transformations to data set, and place in dictionary

data_transformations = transforms.Compose([
    #transforms.Resize(input_shape),
    transforms.ToTensor(),
    transforms.Normalize(mean,std),

])

image_datasets = {x:datasets.ImageFolder(os.path.join(PATH,x),data_transformations)
                  for x in ['train','valid']}

#load data
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True)
               for x in ['train','valid']}



#prepare gpu
device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
device



dataset_sizes = {x:len(image_datasets[x]) for x in ['train','valid']}
#ignore ipython checkpoints
class_names = image_datasets['train'].classes[1:]



model_conv = torchvision.models.densenet161(pretrained=True)



#freeze all layers
for param in model_conv.parameters():
    param.requires_grad = False




model_conv.classifier.in_features



#change final layer for our number of classes; it's also not frozen!
num_ftrs = model_conv.classifier.in_features
#determined by running network on different sized image and seeing how much is extracted
ftrs_from_size = 317952
model_conv.classifier = nn.Linear(ftrs_from_size,len(class_names))
model_conv.classifier



model_conv = model_conv.to(device)



lr           = 1e-4 #default is 1e-3 ; best has been 1e-4
betas        = (0.9, 0.999)
eps          = 1e-08
weight_decay = 0

criterion    = nn.CrossEntropyLoss()

#optimizer takes iterable of parameters to optimize, and then whatever defaults we want specific to training method
#updates these parameters as a we get gradients per minibatch, including lr
optimizer_conv = optim.Adam(list(filter(lambda p: p.requires_grad, model_conv.parameters())) ,lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, amsgrad=False)
optimizer_conv



def train_model(model, criterion, optimizer, num_epochs):

    pred_list  = []
    prob_list  = []
    label_list = []
    loss_list  = {'train':[],'valid':[]}
    acc_list   = {'train':[],'valid':[]}


    soft_max = nn.Softmax(dim=1).to(device)


    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'valid']:
            if phase == 'train':
                #scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode
                prob_temp  = []
                pred_temp  = []
                label_temp = []


            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = (labels-1).to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)



                    if phase=='train':
                        loss.backward()
                        optimizer.step()

                    else:
                        pred_temp.extend(preds.tolist())
                        prob_temp.extend(soft_max(outputs).tolist())
                        label_temp.extend(labels.tolist())



                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc  = running_corrects.double() / dataset_sizes[phase]
            runtime    = (time.time()-since)

            loss_list[phase].append(epoch_loss)
            acc_list[phase].append(epoch_acc)

            print('{} Loss: {:.4f} Acc: {:.4f} Time: {:.4f}'.format(
                phase, epoch_loss, epoch_acc, runtime/60))

            # deep copy the model
            if phase == 'valid' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                prob_list  = prob_temp
                pred_list  = pred_temp
                label_list = label_temp


        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    #remove soft_max from memory
    del soft_max
    return model,pred_list,label_list, prob_list,loss_list,acc_list




mod,_,_, _,loss_list,acc_list = train_model(model_conv, criterion, optimizer_conv, num_epochs=1)


# <h2>Training Entire Network</h2>



for param in mod.parameters():
    param.requires_grad = True


# In[ ]:


#entire network:
optimizer_conv = optim.Adam( mod.parameters() ,lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, amsgrad=False)

mod,pred_list,label_list, prob_list,loss_list2,acc_list2 = train_model(mod, criterion, optimizer_conv, num_epochs=5)







loss_list['train'] = loss_list['train']+loss_list2['train']
loss_list['valid'] = loss_list['valid']+loss_list2['valid']
acc_list['train']  = acc_list['train'] +acc_list2['train']
acc_list['valid']  = acc_list['valid'] +acc_list2['valid']


# <h2>Analyze Results</h2>


def plot_confusion_matrix(cm, classes,
                          normalize=True,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
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



xaxis = np.arange(len(loss_list['train']))
plt.plot(xaxis, loss_list['train'] )
plt.xlabel("epoch")
plt.xticks(xaxis)
plt.ylabel("training loss")
plt.title("training loss over epochs")
plt.show()



xaxis = np.arange(len(loss_list['valid']))
plt.plot(xaxis,loss_list['valid'])
plt.xticks(xaxis)
plt.xlabel("epochs")
plt.ylabel("valid loss")
plt.title("valid loss over epochs")
plt.show()



xaxis = np.arange(len(acc_list['train']))
plt.plot(xaxis,acc_list['train'] )
plt.xticks(xaxis)
plt.xlabel("epoch")
plt.ylabel("training accuracy")
plt.title("training accuracy over epochs")
plt.show()



xaxis = np.arange(len(acc_list['valid']))
plt.plot(xaxis,acc_list['valid'])
plt.xlabel("epochs")
plt.xticks(xaxis)
plt.ylabel("valid accuracy")
plt.title("valid accuracy over epochs")
plt.show()



from sklearn.metrics import confusion_matrix
cm = confusion_matrix(label_list, pred_list)



plot_confusion_matrix(cm, class_names)



n_classes = np.arange(len(class_names))
fpr     = dict()
tpr     = dict()
roc_auc = dict()



y_roc = label_binarize(np.asarray(label_list), classes=n_classes)



for i in n_classes:
    fpr[i],tpr[i],_ = roc_curve(y_roc[:,i],np.asarray(prob_list)[:,i])
    roc_auc[i]      = auc(fpr[i],tpr[i])



plt.figure()
lw = 2
colors = ['b','g','r','c','m','y','k']

for i in n_classes:
    plt.plot(fpr[i], tpr[i], color=colors[i],
         lw=lw, label=class_names[i]+' ROC curve (area = %0.2f)' % roc_auc[i])


plt.plot([0, 1], [0, 1], color='red', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()
