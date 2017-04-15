#**Behavioral Cloning** 
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
[track_1]: ./images/data_1.PNG "Track 1" 	
[track_2]: ./images/data_2.PNG "Track 2" 	

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

**In this link I have summarized my experience along this project: https://www.facebook.com/nacho.sanudo/videos/1428878897171430/
I hope you like it :)**

My project includes the following files:
* train.py containing the script to create and train the model (I would like to separate the file into model.py+train.py, but there is a bug of python using Windows https://bugs.python.org/issue19539 )
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results
* NVIDIA_RUN.mp4 video


####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model_NVIDIA.h5
python drive.py model_comma.h5
```

####3. Submission code is usable and readable

The train.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

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

The overall strategy for deriving a model architecture was to not reinvent the wheel, in order to do so I decided to adopt state of the art techniques to implement in my project, that is, NVIDIA and comma.ai models that are well-known in the community. 

My first tries with these models were so bad, there were a few spots where the vehicle fell off the track, as is detailed in the video :)

With the scope to tackle with these problems and to improve the driving behavior, I applied data augmentation into my dataset, I did a little research about some techniques used in data augmentation, these are: 

- (image_processing.py)

To combat the overfitting, I modified the model adding dropout functions.

I applied two techniques from here https://medium.com/@mohankarthik/cloning-a-car-to-mimic-human-driving-5c2f7e8d8aff

The first thing is related with the steering angle, I detected that the angle distribution was very unbalanced with a big amount of zeros in the distribution, the original distribution is shown here: ![Left][image_distribution]

I also flipped images and angles thinking that this would generalize better the training. For example, here is an image that has then been flipped: ![Left][imageFlipped]

At the end of the process, the vehicle is able to drive autonomously around the track one without leaving the road.

####2. Final Model Architecture

The final model architecture is this one ![Left][imageNVIDIA]

####3. Creation of the Training Set & Training Process

As I mentioned above, I decided to use the Udacity dataset, however, I also created my own dataset following the course tips (for instance, two laps following the right lanes, two the left etc...).

After the collection process, I had this number of images:
Track 1 --> ![Left][track_1]
Track 2 --> ![Left][track_2]

I didin't test it because I am working with my laptop and I would not like to burn my GPU :S
As a future work I would like to train and test the first track with this large dataset and test the driving in the second track. 

This project was a very good experience :)

