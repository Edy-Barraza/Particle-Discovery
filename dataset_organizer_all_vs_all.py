import os
import sys
import shutil
import math
import subprocess as sp
from random import shuffle

"""
- Organizes Image datasets into training|validation|testing for classifying particles represented as images
- For a particle to be considered,it must initially reside inside a folder with its name
  with the other images of that particle, and this folder must be in the directory of PATH
- Particles specified in the categories list will receive a training|validation|testing split
- If "other" is specified in categories list, particles not specified in the categories array
  are considered part of "other" category and receive training|validation|testing split
- If "other" is not specified in categories list, then particles not specified in categories list
  are ignored
- Can choose to balance dataset, meaning that each category specified has an equal number of images
  determined by the class with the lowest amount of images
- If balance is chosen in presence of "other" category, then particles not specified in categories
  list are equally distributed in "other" category
"""


PATH          = "../../data/images_600/all_vs_all/"
categories    = ["other","RelValHiggs200ChargedTaus_13" ]
balanced      = False
train_percent = .6
valid_percent = .2
test_percent  = .2
classes = os.listdir(PATH)
split = ["train/","valid/","test/"]



def create_folders(categories):
    for folder in split[:2]:
        sp.call("mkdir "+PATH+folder,shell=True)
        for category in categories:
            sp.call("mkdir "+PATH+folder+category+"/",shell=True)
    sp.call("mkdir "+PATH+split[2],shell=True)

def split_numbs_help(total, train_percent,valid_percent,test_percent):
    train_len = total*train_percent
    valid_len = total*valid_percent
    test_len  = total*test_percent

    if (total-math.floor(train_len)-math.ceil(valid_len)-math.ceil(test_len)==0):
        return math.floor(train_len),math.ceil(valid_len),math.ceil(test_len)
    else:
        return math.ceil(train_len),math.floor(valid_len),math.floor(test_len)

if balanced:
    smallest = math.inf
    for class_ in classes:
        smallest =  min(smallest,len(os.listdir(PATH+class_)))
    train_numb,valid_numb,test_numb = split_numbs_help(smallest,train_percent,valid_percent,test_percent)





def partition_class(class_,category,train_numb,valid_numb,test_numb):
    files = os.listdir(PATH+class_)
    shuffle(files)
    for f in files:
        if train_numb>0:
            shutil.move(PATH+class_+"/"+f,PATH+'train/'+category+"/"+f)
            train_numb-=1
        elif valid_numb>0:
            shutil.move(PATH+class_+"/"+f,PATH+'valid/'+category+"/"+f)
            valid_numb-=1
        elif test_numb>0:
            shutil.move(PATH+class_+"/"+f,PATH+'test/'+f)
            test_numb-=1
        else:
            break



print("Organizing DataSet")
create_folders(categories)
print("Created Folders")
if balanced:
    #calculating smallest file numbers for balanced dataset
    smallest = math.inf
    for class_ in classes:
        smallest =  min(smallest,len(os.listdir(PATH+class_)))
    train_numb,valid_numb,test_numb = split_numbs_help(smallest,train_percent,valid_percent,test_percent)

    #determining if other is in dataset so we can have balanced other dataset
    divisor = sum(class_ not in categories for class_ in classes)
    for class_ in classes:
        if class_ in categories:
            partition_class(class_,class_,train_numb,valid_numb,test_numb)
        else:
            partition_class(class_,"other",train_numb/divisor,valid_numb/divisor,test_numb/divisor)

        print("Organized "+class_)
        print(class_+" has "+str(len(os.listdir(PATH + class_))) +" files left after organizing")

if !balanced:
    for class_ in classes:
        total_files = len(os.listdir(PATH+class_))
        train_numb,valid_numb,test_numb = split_numbs_help(total_files,train_percent,valid_percent,test_percent)
        if class_ in categories:
            partition_class(class_,class_,train_numb,valid_numb,test_numb)
        else:
            partition_class(class_,"other",train_numb,valid_numb,test_numb)

        print("Organized "+class_)
        print(class_+" has "+str(len(os.listdir(PATH + class_))) +" files left after organizing")

for class_ in classes:
    sp.call("rm -r "+PATH+class_ , shell=True)
