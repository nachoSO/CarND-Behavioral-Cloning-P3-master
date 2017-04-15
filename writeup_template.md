#**Behavioral Cloning** 

##Writeup Template
---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image_distribution]: ./images/figure_distribution.png "Steering distribution"
[imageNVIDIA]: ./images/nvidia_architecture.PNG "NVIDIA architecture" 	
[imageFlipped]: ./images/flip.PNG "Flipped image" 	

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* train.py containing the script to create and train the model (I would like to separate the file into model.py+train.py, but there is a bug of python using Windows https://bugs.python.org/issue19539 )
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results

####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model_NVIDIA.h5
python drive.py model_comma.h5
```

####3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed
I've implemented two models:

- NVIDIA (https://arxiv.org/abs/1604.07316)
```
 model = Sequential()
 model.add(Lambda(lambda x: x/127.5 - 1.,
        input_shape=(row, col,ch),
        output_shape=(row, col,ch)))
 model.add(Cropping2D(cropping=((70,25),(0,0))))
 model.add(Dropout(.1))
 model.add(Convolution2D(24,5,5, subsample=(2,2),activation="relu"))
 model.add(Dropout(.2))
 model.add(Convolution2D(36,5,5, subsample=(2,2),activation="relu"))
 model.add(Dropout(.2))
 model.add(Convolution2D(48,5,5, subsample=(2,2),activation="relu"))
 model.add(Dropout(.2))
 model.add(Convolution2D(64,3,3, subsample=(2,2),activation="relu"))
 model.add(Dropout(.5))
 model.add(Flatten())
 model.add(Dense(100))
 model.add(ELU())
 model.add(Dense(50))
 model.add(ELU())
 model.add(Dense(10))
 model.add(ELU())
 model.add(Dense(1))
```
- comma.ai (https://github.com/commaai/research/blob/master/train_steering_model.py)
```
 Cropping2D(cropping=((70,25),(0,0)), input_shape=(160,320,3))
 Lambda(lambda x: x/255.0 - 0.5)
 Convolution2D(16, 8, 8, subsample=(4, 4), border_mode="same")
 ELU()
 Convolution2D(32, 5, 5, subsample=(2, 2), border_mode="same")
 ELU()
 Convolution2D(64, 5, 5, subsample=(2, 2), border_mode="same")
 Flatten()
 Dropout(.2)
 ELU()
 Dense(512)
 Dropout(.5)
 ELU()
 Dense(1)
```
**For the sake of simplicity I will detailed my implementation of the NVIDIA model for the rest of the report, because, the comma.ai model was full taken from the comma.ai github with only just one modification (I added a cropping layer).**

####2. Attempts to reduce overfitting in the model

Dropout layers are state of the art method to reduce the overfitting as is detailed here (https://www.cs.toronto.edu/~hinton/absps/JMLRdropout.pdf). In order to reduce the overfitting I added dropouts functions after each convolution in the NVIDIA model (train.py nvidia_model function)

####3. Model parameter tuning

The model used an adam optimizer, using the same configuration detailed in the course.

####4. Appropriate training data

I've created a large dataset however, I am using my laptop's GPU (850M), so, for the sake of simplicity in terms of time, I've decided to use only the Udacity dataset, maybe in the future as challenge, i'll train the entire dataset (2GB more or less) with a decent GPU.

###Model Architecture and Training Strategy

####1. Solution Design Approach

The overall strategy for deriving a model architecture was to ...

My first step was to use a convolution neural network model similar to the ... I thought this model might be appropriate because ...

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model so that ...

Then I ... 

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track... to improve the driving behavior in these cases, I ....

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

####2. Final Model Architecture

The final model architecture is this one ![Right][imageNVIDIA]

####3. Creation of the Training Set & Training Process

As I've mentioned above, I've decided to use the Udacity dataset, however, I've also created my own dataset following the course tips (for instance, two laps following the right lanes, two the left etc...)

To augment the data sat, I also flipped images and angles thinking that this would generalize better the training. For example, here is an image that has then been flipped: [imageFlipped]

After the collection process, I had X number of data points. I then preprocessed this data by ...


I finally randomly shuffled the data set and put Y% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was Z as evidenced by ... I used an adam optimizer so that manually training the learning rate wasn't necessary.
