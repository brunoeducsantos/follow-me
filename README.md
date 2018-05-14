# Follow Me

## Overview
In this project, you will train a deep neural network to identify and track a target in simulation. So-called “follow me” applications like this are key to many fields of robotics and the very same techniques you apply here could be extended to scenarios like advanced cruise control in autonomous vehicles or human-robot collaboration in industry.

[image_0]: https://github.com/BrunoEduardoCSantos/Follow-Me/blob/master/imgs/follow%20me.PNG
![alt text][image_0] 
## Project Files

My project includes the following files:
*  [model_training](https://github.com/BrunoEduardoCSantos/Follow-Me/blob/master/code/model_training.ipynb): containing the script to create and train the model
* [preprocess_ims](https://github.com/BrunoEduardoCSantos/Follow-Me/blob/master/code/preprocess_ims.py): script for pre-process images to feed the model
* [follower](https://github.com/BrunoEduardoCSantos/Follow-Me/blob/master/code/follower.py): script to activate drone's following mode based on created model to identify people on images
* [model_weights](https://github.com/BrunoEduardoCSantos/Follow-Me/blob/master/data/weights/model_weights): h5 file containing the weights of the model

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

## Step by step running code procedure

Using the Udacity simulator and my #follower.py# file, the car can be driven autonomously around the track by executing 
```sh
python follower.py model_weights.h5
```

## Model Architecture and Training Strategy ##
The model is build on three parts:
* Encoder
* 1x1 convolution
* Decoder

The encoder is composed by 3 convolution layers described as follows:
* 3x3 receptive filter per layer 
* Depth = {32; 64 ;128}
* 2X2 stride
* Batch normalization

The 1X1 convolution is a convolution layer with the following properties:
* keeps spatial information 
* adds non-linearity if the depth is the same of previous convolution layer
* allows to combine weights from different depth layer with a similar effect of a fully connected layer
* Allows to reduce the number of parameters and reduce computational time
For the chosen architecture the goal was reducing the computational cost as well as keep the spatial information. For instance, in this architecture the number of filters or depth was 32.

Finally, the decoder is composed by 2x upsampling layers followed by convolution + batchnormalization and skip connections with encoder layers to improve lost spatial features resolution.
The decoder is composed by the following layers:
* Upsampling layers ( bilinear interpolation)
* Convolution Layers (Depth= {32,64,128},3X3 receptive filter, 2x2 stride)
* Skip connections

By upsampling to desired size will be possible to calculate the pixel values at each point using a interpolation method such as bilinear interpolation.
Convolution layers allow to choose relevant features on images to identify objects in a image.
Finally, skip connections allow to provide information lost on encoder layers by increasing spatial resolution.

### Solution Design Approach
Firstly, the use of encoder and decoders to apply segmentation of objects in a image is based on pixel by pixel learning instead of image invariance filters as used in image classification where the spatial information is not so relevant.
The overall strategy for deriving a model architecture began with a base on initial convolution layer of depth 32 with 3x3 filter , 1x1 convolution with depth 8  and decoder with same depth than encoder. The reason for this start was based on image input size 256X256X3. 
From this point, several convolution layers were added with increasing depth (based on powers of 2). 
This approach was based on SegNet architecture used by Stanford to segment objects in a image.
It is important to mention that the 1x1 layer depth increase was correlated with data generation to reduce overfitting and  model performance improvement. The data generation was important to reduce the error (cross-entropy) of training and validation datasets as well overcome the local minimum and allow the netowork to continue learning. 

#### Hyperparameters
The learning rate was selected based on a  manual decay related with :
* training dataset error 
* rise of data generated
* overcome local minimum

The range of learning rate was: 0.01-0.0009. 


Regarding the batch_size it was calculated based on initial dataset size of 7100 images by estimating around 120 steps_per_epoch. Therefore, the batch_size was kept equal to 64. Another reason behind this value is save computation time to train the nework.  Eventually, this number could be increased in order to avoid floatuation of error through epochs.


The chosen number of epochs was 80. The adopted procedure was recording 15 epochs each time and save the weights according to error keep decreasing and the network could converge to a local minimum.


### Final architecture
The final architecture visualization is given by: 

[image1]: https://github.com/BrunoEduardoCSantos/Follow-Me/blob/master/imgs/FollowMeArchitecture.png "Model Visualization"
![alt text][image1] 

If the model was generalize for segment other objects this model could be as a base since it already learned basic shapes such as circles , rectangle. Hence, it could be used in a transfer model process to generalize to other objects segmentation.

## Scoring ##

To score the network on the Follow Me task, two types of error are measured. First the intersection over the union for the pixelwise classifications is computed for the target channel. 

In addition to this we determine whether the network detected the target person or not. If more then 3 pixels have probability greater then 0.5 of being the target person then this counts as the network guessing the target is in the image. 

We determine whether the target is actually in the image by whether there are more then 3 pixels containing the target in the label mask. 

Using the above the number of detection true_positives, false positives, false negatives are counted. 

**How the Final score is Calculated**

The final score is the pixelwise `average_IoU*(n_true_positive/(n_true_positive+n_false_positive+n_false_negative))` on data similar to that provided in sample_evaulation_data.
The model average IoU was : 0.49.

