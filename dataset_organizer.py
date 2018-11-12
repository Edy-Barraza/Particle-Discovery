import os
import sys
import shutil
import math
import subprocess as sp
from random import shuffle
import argparse

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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--PATH",type=str,required=True,help="(str) path to folder with unorganized dataset")
    parser.add_argument("--train_percent",type=float,required=True,help="(int) percentage of data to allocate to training set")
    parser.add_argument("--valid_percent",type=float,required=True,help="(int) percentage of data to allocate to validation set")
    parser.add_argument("--test_percent",type=float,required=True,help="(int) percentage of data to allocate to testing set")
    parser.add_argument("--balanced",action="store_true",help="(flag) optional, whether to make dataset balanced with equal number of samples in each class")
    parser.add_argument("--delete_excess",action="store_true",help="(flag) optional, whether to delete class folders after moving necessary files")
    parser.add_argument("--categories",nargs="+",type=str,help="(n strings) optional, strings of folder names we will consider a class when organizig data")

    args = parser.parse_args()

    """
    Args:
        PATH (str) - path to folder with unorganized dataset
        train_percent (int) - percentage of data to allocate to training set
        valid_percent (int) - percentage of data to allocate to validaiton set
        test_percent (int) - percentage of data to allocate to testing set
        balanced (flag) - optional, whether to make dataset balanced with equal number of samples in each class
        delete_excess (flag) - optional, whether to delete class folders after moving necessary files
        categories (n strings) - optional, strings of folder names we will consider a class when organizig data
    """

    categories = os.listdir(args.PATH)
    classes    = os.listdir(args.PATH)
    if args.categories:
        categories = args.categories


    print("Organizing DataSet")
    create_folders(categories,args.PATH)
    print("Created Folders")
    if args.balanced==True:
        #calculating smallest file numbers for balanced dataset
        smallest = math.inf
        for class_ in classes:
            smallest =  min(smallest,len(os.listdir(args.PATH+class_)))
        train_numb,valid_numb,test_numb = split_numbs_help(smallest,args.train_percent,args.valid_percent,args.test_percent)

        #determining if other is in dataset so we can have balanced other dataset
        divisor = sum(class_ not in categories for class_ in classes)
        for class_ in classes:
            if class_ in categories:
                partition_class(class_,class_,train_numb,valid_numb,test_numb,args.PATH)
            if "other" in categories and (class_ not in categories):
                partition_class(class_,"other",train_numb/divisor,valid_numb/divisor,test_numb/divisor,args.PATH)

            print("Organized "+class_)
            print(class_+" has "+str(len(os.listdir(args.PATH + class_))) +" files left after organizing")

    if args.balanced!=True:
        for class_ in classes:
            total_files = len(os.listdir(args.PATH+class_))
            train_numb,valid_numb,test_numb = split_numbs_help(total_files,args.train_percent,args.valid_percent,args.test_percent)
            if class_ in categories:
                partition_class(class_,class_,train_numb,valid_numb,test_numb,args.PATH)
            if "other" in categories and (class_ not in categories):
                partition_class(class_,"other",train_numb,valid_numb,test_numb,args.PATH)

            print("Organized "+class_)
            print(class_+" has "+str(len(os.listdir(args.PATH + class_))) +" files left after organizing")

    if args.delete_excess:
        for class_ in classes:
            sp.call("rm -r "+args.PATH+class_ , shell=True)




split = ["train/","valid/","test/"]
def create_folders(categories,PATH):
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


def partition_class(class_,category,train_numb,valid_numb,test_numb,PATH):
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

if __name__=='__main__':
    main()
