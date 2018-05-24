# Follow Me

## Overview
In this project, a deep neural network was used to identify and track a target in simulation. So-called “follow me” applications like this are key to many fields of robotics and the very same techniques you apply here could be extended to scenarios like advanced cruise control in autonomous vehicles or human-robot collaboration in industry.

[image_0]: https://github.com/BrunoEduardoCSantos/Follow-Me/blob/master/imgs/follow%20me.PNG
![alt text][image_0] 

## Project Files

The project includes the following files:
*  [model_training](https://github.com/BrunoEduardoCSantos/Follow-Me/blob/master/code/model_training.ipynb): containing the script to create and train the model
* [preprocess_ims](https://github.com/BrunoEduardoCSantos/Follow-Me/blob/master/code/preprocess_ims.py): script for pre-process images to feed the model
* [follower](https://github.com/BrunoEduardoCSantos/Follow-Me/blob/master/code/follower.py): script to activate drone's following mode based on created model to identify people on images
* [model_weights](https://github.com/BrunoEduardoCSantos/Follow-Me/blob/master/data/weights/model_weights): h5 file containing the weights of the model

## Setup Instructions
**Clone the repository**
```
$ git clone https://github.com/BrunoEduardoCSantos/Follow-Me.git
```
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
A simple training dataset has been provided in this project's repository. This dataset will allow you to verify that the segmentation network is semi-functional. However, if your interested in improving your score,you may want to collect additional training data. To do it, please see the following steps:
* Download the simulator [here](https://github.com/udacity/RoboND-DeepLearning/releases/latest)
* Create  data directory is organized as follows:
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
* Collect data from the simulator using training mode

### Image Preprocessing ###
Before the network is trained, the images first need to be undergo a preprocessing step. The preprocessing step transforms the depth masks from the sim, into binary masks suitable for training a neural network. It also converts the images from .png to .jpeg to create a reduced sized dataset, suitable for uploading to AWS. 
To run preprocessing:
```
$ python preprocess_ims.py
```
After pre-processing the raw images ,we obtained the processed images as well as the masks. On the following picture is possible visualize a ground truth image (i.e. processed image with mask).

[image2]:  https://github.com/BrunoEduardoCSantos/Follow-Me/blob/master/imgs/download.png "Ground truth"
![alt text][image2] 

## Step by step running code procedure

Using the Udacity simulator and my *follower.py* file, the car can be driven autonomously around the track by executing 
```sh
python follower.py model_weights.h5
```
After launch the simulator in autonomous mode, which combined with the previous script will allows the Quad to follows autonomously the "hero".

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
Neurons in a fully connected layer have full connections to all activations in the previous layer, as seen in regular Neural Networks. Their activations can hence be computed with a matrix multiplication followed by a bias offset. It is worth noting that the difference between FC and CONV layers is that the neurons in the CONV layer are connected only to a local region in the input, and that many of the neurons in a CONV volume share parameters. 

Finally, the decoder is composed by bilinear upsampling layers followed by convolution + batch-normalization and skip connections with encoder layers to improve lost of spatial features resolution.

The decoder is composed by the following layers:

* Upsampling layers ( bilinear interpolation)
* Convolution Layers (Depth= {32,64,128},3X3 receptive filter, 2x2 stride)
* Skip connections

Bilinear upsampling is a resampling technique that utilizes the weighted average of four nearest known pixels, located diagonally to a given pixel, to estimate a new pixel intensity value.

Including convolution layers allow to choose relevant features on images to identify objects in a image.

Finally, skip connections allow to provide information lost on encoder layers by increasing spatial resolution. This method is important to the decoder since the upsampling doesn’t recover all the spatial information,i.e, the interpolation process need more information from the input image to capture more spatial resolution.

The final architecture visualization is given by: 

[image1]: https://github.com/BrunoEduardoCSantos/Follow-Me/blob/master/imgs/FollowMeArchitecture.png "Model Visualization"
![alt text][image1] 


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


## Generalization to other object class

From the created model based on fully convolution network it was created a remarkable ability of learning high-level representations
for object recognition by learning high level object contours. 

Since people and dogs/cats have similar high level contours than humans, using transfer learning it would possible increasing encoder/decoder layer depth to detect animals in the images.  



## Scoring ##

To score the network on the Follow Me task, two types of error are measured. First the intersection over the union for the pixelwise classifications is computed for the target channel. 

In addition to this we determine whether the network detected the target person or not. If more then 3 pixels have probability greater then 0.5 of being the target person then this counts as the network guessing the target is in the image. 

We determine whether the target is actually in the image by whether there are more then 3 pixels containing the target in the label mask. 

Using the above the number of detection true_positives, false positives, false negatives are counted. 

**How the Final score is Calculated**

The final score is the pixelwise `average_IoU*(n_true_positive/(n_true_positive+n_false_positive+n_false_negative))` on data similar to that provided in sample_evaulation_data.
The model average IoU was : 0.49.

For better visual evaluation about the performance of the model, in the following pictures there are resulting predictions compared with ground truth as well as original images.

[image3]:https://github.com/BrunoEduardoCSantos/Follow-Me/blob/master/imgs/hero.png "Hero prediction"
![alt text][image3] 

[image4]:https://github.com/BrunoEduardoCSantos/Follow-Me/blob/master/imgs/patrol.png "People prediction"
![alt text][image4] 

[image5]:https://github.com/BrunoEduardoCSantos/Follow-Me/blob/master/imgs/people.png "People prediction"
![alt text][image5] 

Order of images from left to right: RGB images (left) ; Image with masks / Ground truth (mid) ; Model prediction (right)



## Disclamer
This project was cloned from [Udacity deep learning project](https://github.com/udacity/RoboND-DeepLearning-Project) in the context of [Robotics Software Engineer nanodegree](https://www.udacity.com/course/robotics-software-engineer--nd209).

