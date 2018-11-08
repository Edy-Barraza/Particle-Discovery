from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
from torchvision import models, transforms, datasets
import numpy as np

import argparse
import time
import copy
import sys
import os

import analysis


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bs',type=int,default=32)
    parser.add_argument('--epochs_last_layer',type=int,default=0)
    parser.add_argument('--epochs',type=int,default=10)
    parser.add_argument('--shape',type=int,default=224)
    parser.add_argument('--lr',type=float,default=1e-4)
    parser.add_argument('--betas',nargs=2,type=float)
    parser.add_argument('--weight_decay',type=float,default=0.0)
    parser.add_argument('--eps',type=float,default=1e-8)
    parser.add_argument('--dropout',type=float,default=0)
    parser.add_argument('--fc_features',type =int,default=2208)
    parser.add_argument('--means',nargs=3,type=float)
    parser.add_argument('--stdevs',nargs=3,type=float)
    parser.add_argument('--pretrained',action="store_true")
    parser.add_argument('--save_model', action="store_true")
    parser.add_argument('--PATH_model_save',type=str)
    parser.add_argument('--PATH_data',type=str)
    parser.add_argument('--PATH_save_images',type=str)

    args = parser.parse_args()

    """
    Args:
        bs (int) - batch size
        epcohs_last_layer (int) - # of epochs to train only the fully connected layer of net
        epochs (int) - # of epochs to train whole neural network
        shape (int) - height in pixels for resizing square image
        lr (float) - learning late for ADAM optimizer
        betas ( 2 floats) - betas for ADAM optimizer
        weight_decay (float) - weight decay
        eps (float) - epsilon term added learning rate denominator for numerical stability
        dropout (float) - drop out rate for dropout layers of network
        means (3 floats) - mean pixel values for R,G, & B color channels
        stdevs (3 floats) - std dev pixel values for R,G, & B color channels
        pretrained (flag) - designates if we should start with pretrained weights before training
        fc_features (int) - number of features extracted by network that's fed to fully connected layer
        save_model (flag) - designates if we should save the model
        PATH_model_save (str) - path for saving model
        PATH_data (str) - path for accessing data folder
        PATH_save_images (str) - path to save images of our analysis
    """

    #load & preprocess data
    mean = args.means #[0.485, 0.456, 0.406]
    std  = args.stdevs#[0.229, 0.224, 0.225]

    data_transformations = transforms.Compose([
        transforms.Resize(args.shape),
        transforms.ToTensor(),
        transforms.Normalize(mean,std),
    ])
    image_datasets = {x:datasets.ImageFolder(os.path.join(args.PATH_data,x),data_transformations)
                      for x in ['train','valid']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=args.bs, shuffle=True)
                   for x in ['train','valid']}

    dataset_sizes = {x:len(image_datasets[x]) for x in ['train','valid']}

    class_names = image_datasets['train'].classes

    #device for training net
    device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
    print("Using following device for computation:")
    print(device)

    #prepare model
    mod = torchvision.models.densenet161(pretrained=(args.pretrained ==True),drop_rate=args.dropout)
    if args.epochs_last_layer==0:
        for param in mod.parameters():
            param.requires_grad = False


    """
    must determine # of features by running network
    and getting size mismatch error at fully conected layer
    """
    mod.classifier = nn.Linear(args.fc_features,len(class_names))
    mod = mod.to(device)

    #training characteristics
    optimizer = optim.Adam(list(filter(lambda p: p.requires_grad, mod.parameters())) ,lr=args.lr, betas=tuple(args.betas), eps=args.eps, weight_decay=args.weight_decay, amsgrad=False)
    criterion = nn.CrossEntropyLoss()

    #training
    if args.epochs_last_layer>0:
        mod,loss_list_pre,acc_list_pre = train_model_final_layer(mod, dataloaders, criterion, optimizer, args.epochs_last_layer,dataset_sizes,device)
        for param in mod.parameters():
            param.requires_grad = True
        optimizer = optim.Adam(list(filter(lambda p: p.requires_grad, mod.parameters())) ,lr=args.lr, betas=tuple(args.betas), eps=args.eps, weight_decay=args.weight_decay, amsgrad=False)

    mod,pred_list,label_list, prob_list,loss_list,acc_list =train_model(mod, dataloaders, criterion, optimizer, args.epochs,dataset_sizes,device)

    #save model
    if args.save_model:
        torch.save(mod.state_dict(),args.PATH_model_save)
    #analyze results
    if args.epochs_last_layer>0:
        loss_list['train'] = loss_list_pre['train']+loss_list['train']
        loss_list['valid'] = loss_list_pre['valid']+loss_list['valid']
        acc_list['train']  = acc_list_pre['train'] +acc_list['train']
        acc_list['valid']  = acc_list_pre['valid'] +acc_list['valid']

    analysis.plot_epochs(loss_list['train'],"training loss"     ,args.PATH_save_images)
    analysis.plot_epochs(loss_list['valid'],"validation loss"   ,args.PATH_save_images)
    analysis.plot_epochs(acc_list['train'],"training accuracy"  ,args.PATH_save_images)
    analysis.plot_epochs(acc_list['valid'],"validation accuracy",args.PATH_save_images)
    analysis.plot_confusion_matrix(label_list, pred_list, class_names,args.PATH_save_images)
    analysis.plot_roc(label_list,prob_list,class_names,args.PATH_save_images)

def train_model(model, dataloaders, criterion, optimizer, num_epochs,dataset_sizes,device):

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
                labels = (labels).to(device)

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

def train_model_final_layer(model, dataloaders, criterion, optimizer, num_epochs,dataset_sizes,device):

    loss_list  = {'train':[],'valid':[]}
    acc_list   = {'train':[],'valid':[]}

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



            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = (labels).to(device)

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


        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)

    return model,loss_list,acc_list

if __name__=='__main__':
    main()
