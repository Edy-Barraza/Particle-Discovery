# Particle-Discovery

Everything you need to start categorizing subatomic particles! 
<br>
This repository contains:
<ol type="I">
    <b>
    <li> Dataset Organizer </li>
    <li> Data Normalization </li>
    <li> Neural Network Trainer </li>
    </b>
</ol>
<h3> I. Dataset Organizer </h3>

First, we will organize a dataset into training, validation, and testing set (skip if your data is already split as such). We take a dataset with a file structure as follows:
```
data
├── higgs 
├── jpsi
├── upsilon
└── etc
    └── event.png
```
and use dataset_organizer.py to perform the split for us! For example, we could run the following command in the shell:
```
python dataset_organizer.py --PATH /path/to/unorganized/dataset/ --train_percent .6 --valid_percent .2 --test_percent .2 --categories RelValHiggs200ChargedTaus_13 other
```
dataset_organizer.py has the following arguments: 
```
Args:
        PATH (str) - path to folder with unorganized dataset
        train_percent (int) - percentage of data to allocate to training set
        valid_percent (int) - percentage of data to allocate to validaiton set
        test_percent (int) - percentage of data to allocate to testing set
        balanced (flag) - optional, whether to make dataset balanced with equal number of samples in each class
        delete_excess (flag) - optional, whether to delete class folders after moving necessary files
        categories (n strings) - optional, strings of folder names we will consider as a class when organizig data. If not included, every folder name will be considered a class we organize. If included, folders named will be considered a class we organize; folders not named will be put into "other" if "other" is named, or just ignored  
```

<h3> II. Data Normalization </h3>

After organizing the data set, we need to know the mean and standard deviations for the RGB color channels of our dataset. 
We can use compute_normalization.py for this task! compute_normalization.py will print the means and stdevs for us in the terminal. For example, we could run the following command in the shell:
```
python compute_normalization.py --PATH /path/to/organized/dataset/ --shape 224 
```
compute_normalizations.py has the following arguments:
```
Args:
        PATH (str) - path to folder with data organized into training, validation, testing
        shape (int) - height in pixels for resizing square image. If we don't wish to resize our image, just input original height of image
        bs (int, default:1) - batch size for these computations
        max_samples (int, default:inf) - optional, max number of samples to consider when computing mean & stdev. If dataset is too large to compute in it's entirety, we can pass a maximum number of samples to consider to get an accurate estimation of the mean & stdevs
```
compute_normalization.py will print out the means and the standard deviations, 

<h3> Neural Network Trainer </h3>

Now that we have the means and std devs, we can pass them on to train.py, where we will train a Densely Connected 161 layer Neural Network using the ADAM optimizer. To do so, we run the following command:

```
python train.py --PATH_data /path/to/organized/dataset/ --PATH_save_images /path/to/convenient/folder/ --means 0.002886 0.015588 0.016239 --stdevs 0.052924 0.123025 0.125617 --fc_features 19872
```

train.py run configurations 
```
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
        pretrained (flag) - designates if we should start with Imagenet pretrained weights before training
        save_model (flag) - designates if we should save the model
        PATH_model_save (str) - path for saving model
        PATH_data (str) - path for accessing data folder
        PATH_save_images (str) - path to save images of our analysis     
```

RuntimeError: size mismatch, m1: [4 x 19872], m2: [2208 x 3] at /opt/conda/conda-bld/pytorch_1524586445097/work/aten/src/THC/generic/THCTensorMathBlas.cu:249
