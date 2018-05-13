# Follow Me

## Overview
In this project, you will train a deep neural network to identify and track a target in simulation. So-called “follow me” applications like this are key to many fields of robotics and the very same techniques you apply here could be extended to scenarios like advanced cruise control in autonomous vehicles or human-robot collaboration in industry.

[image_0]: https://github.com/BrunoEduardoCSantos/Follow-Me/blob/master/imgs/follow%20me.PNG
![alt text][image_0] 

## Setup Instructions
**Clone the repository**
```
$ git clone https://github.com/BrunoEduardoCSantos/Follow-Me.git
```

**Download the data**

Save the following three files into the data folder of the cloned repository. 

[Training Data](https://s3-us-west-1.amazonaws.com/udacity-robotics/Deep+Learning+Data/Lab/train.zip) 

[Validation Data](https://s3-us-west-1.amazonaws.com/udacity-robotics/Deep+Learning+Data/Lab/validation.zip)

[Sample Evaluation Data](https://s3-us-west-1.amazonaws.com/udacity-robotics/Deep+Learning+Data/Project/sample_evaluation_data.zip)

**Download the QuadSim binary**

To interface your neural net with the QuadSim simulator, you must use a version QuadSim that has been custom tailored for this project. The previous version that you might have used for the Controls lab will not work.

The simulator binary can be downloaded [here](https://github.com/udacity/RoboND-DeepLearning/releases/latest)

**Install Dependencies**

You'll need Python 3 and Jupyter Notebooks installed to do this project.  The best way to get setup with these if you are not already is to use Anaconda following along with the [RoboND-Python-Starterkit](https://github.com/udacity/RoboND-Python-StarterKit).

If for some reason you choose not to use Anaconda, you must install the following frameworks and packages on your system:
* Python 3.x
* Tensorflow 1.2.1
* NumPy 1.11
* SciPy 0.17.0
* eventlet 
* Flask
* h5py
* PIL
* python-socketio
* scikit-image
* transforms3d
* PyQt4/Pyqt5



## Collecting Training Data ##
A simple training dataset has been provided in this project's repository. This dataset will allow you to verify that your segmentation network is semi-functional. However, if your interested in improving your score,you may want to collect additional training data. To do it, please see the following steps.

The data directory is organized as follows:
```
data/runs - contains the results of prediction runs
data/train/images - contains images for the training set
data/train/masks - contains masked (labeled) images for the training set
data/validation/images - contains images for the validation set
data/validation/masks - contains masked (labeled) images for the validation set
data/weights - contains trained TensorFlow models

data/raw_sim_data/train/run1
data/raw_sim_data/validation/run1
```
### Image Preprocessing ###
Before the network is trained, the images first need to be undergo a preprocessing step. The preprocessing step transforms the depth masks from the sim, into binary masks suitable for training a neural network. It also converts the images from .png to .jpeg to create a reduced sized dataset, suitable for uploading to AWS. 
To run preprocessing:
```
$ python preprocess_ims.py
```

## Model Architecture and Training Strategy ##
The model is build on three parts:
* Encoder
* 1x1 convolution
* Decoder

The encoder is composed by 3x3 convolution layers with 32, 64 and 128 depth joined with a batch normalization. The stride used in each convolution layer is 2x2. 
The 1X1 convolution is a convolution layer that keeps the spatial information by adding non-linearity and a new depth. For instance, for this architecture a depth of 32.
Finally, the decoder is composed by 2x upsampling layers followed by convolution + batchnormalization and skip connections with encoder layers to improve lost spatial features resolution.

### Solution Design Approach

The overall strategy for deriving a model architecture began with a base on initial convolution layer of depth 32 with 3x3 filter , 1x1 convolution with depth 8  and decoder with same depth than encoder. The reason for this start was based on image input size 256X256X3. 
From this point, several convolution layers were added with increasing depth (based on powers of 2). 
This approach was based on SegNet architecture used by Stanford to segment objects in a image.
It is important to mention that the 1x1 layer depth increase was correlated with data generation to reduce overfitting and  model performance improvement. The data generation was important to reduce the error (cross-entropy) of training and validation datasets as well overcome the local minimum and allow the netowork to continue learning. 
In addition, to allow a gradual learning and to avoid overcome the global minimum, it was used a learning rate manual decay from 0.01 to 0.0009 as the error decreases to lower values ( 0.001 magnitude). The others hyperparameters were:
* batch_size: 64
* num_epochs:80
* steps_per_epoch : 119
The chosen batchsize was choosen based on performance so that the network could learn faster.

### Final architecture
The final architecture visualization is given by: 
[image_1]: https://github.com/BrunoEduardoCSantos/Follow-Me/blob/master/imgs/FollowMeArchitecture.png
![alt text][image_1] 



## Scoring ##

To score the network on the Follow Me task, two types of error are measured. First the intersection over the union for the pixelwise classifications is computed for the target channel. 

In addition to this we determine whether the network detected the target person or not. If more then 3 pixels have probability greater then 0.5 of being the target person then this counts as the network guessing the target is in the image. 

We determine whether the target is actually in the image by whether there are more then 3 pixels containing the target in the label mask. 

Using the above the number of detection true_positives, false positives, false negatives are counted. 

**How the Final score is Calculated**

The final score is the pixelwise `average_IoU*(n_true_positive/(n_true_positive+n_false_positive+n_false_negative))` on data similar to that provided in sample_evaulation_data

