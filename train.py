import os
import csv
import image_processing as ip
import tensorflow as tf
import cv2
import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Lambda
from keras.models import Sequential
from keras.layers import Input, Dense, Activation, Flatten, Dropout, ELU
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.models import Sequential, Model
from keras.layers import Cropping2D

def nvidia_model(ch,row,col):
    model = Sequential()
    model.add(Lambda(lambda x: x/127.5 - 1.,
            input_shape=(row, col,ch),
            output_shape=(row, col,ch)))
    model.add(Cropping2D(cropping=((70,25),(0,0))))
    model.add(Convolution2D(24,5,5, subsample=(2,2),activation="relu"))
    model.add(Convolution2D(36,5,5, subsample=(2,2),activation="relu"))
    model.add(Convolution2D(48,5,5, subsample=(2,2),activation="relu"))
    model.add(Convolution2D(64,3,3, subsample=(2,2),activation="relu"))
    model.add(Dropout(.2))
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dropout(.2))
    model.add(Dense(10))
    model.add(Dense(1))
    model.compile(optimizer="adam", loss="mse")

    return model
 
   
def comma_model(ch,row,col):
    model = Sequential()

    model.add(Cropping2D(cropping=((70,25),(0,0)), input_shape=(160,320,3)))
    model.add(Lambda(lambda x: x/255.0 - 0.5))
    model.add(Convolution2D(16, 8, 8, subsample=(4, 4), border_mode="same"))
    model.add(ELU())
    model.add(Convolution2D(32, 5, 5, subsample=(2, 2), border_mode="same"))
    model.add(ELU())
    model.add(Convolution2D(64, 5, 5, subsample=(2, 2), border_mode="same"))
    model.add(Flatten())
    model.add(Dropout(.2))
    model.add(ELU())
    model.add(Dense(512))
    model.add(Dropout(.5))
    model.add(ELU())
    model.add(Dense(1))

    model.compile(optimizer="adam", loss="mse")

    return model
    
def plot_distribution(train_data):
    steering = np.float32(np.array(train_data)[:, 3])
    plt.title('Steering distribution')
    plt.hist(steering, 75)
    plt.ylabel('Number images'), plt.xlabel('steering angle')
    plt.show()

    
def main(_):
    samples = []
            
    # with open('./data2/driving_log.csv') as csvfile:
        # reader = csv.reader(csvfile)
        # for line in reader:
            # samples.append(line)
            
    with open('./data_udacity/driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            samples.append(line)
            
    from sklearn.model_selection import train_test_split
    train_samples, validation_samples = train_test_split(samples, test_size=0.2)
    #plot_distribution(train_samples)

    train_generator = ip.generator(train_samples, batch_size=64)
    validation_generator = ip.generator(validation_samples, batch_size=64)
    ch, row, col = 3, 160, 320  # camera format
    model=comma_model(ch,row,col)

    history_object=model.fit_generator(train_generator, steps_per_epoch= 
                len(train_samples), validation_data=validation_generator, 
                validation_steps=len(validation_samples), epochs=32)
                
    # ### print the keys contained in the history object
    # print(history_object.history.keys())
    # ### plot the training and validation loss for each epoch
    # plt.plot(history_object.history['loss'])
    # plt.plot(history_object.history['val_loss'])
    # plt.title('model mean squared error loss')
    # plt.ylabel('mean squared error loss')
    # plt.xlabel('epoch')
    # plt.legend(['training set', 'validation set'], loc='upper right')
    # plt.show()

    model.save('model.h5')

if __name__ == '__main__':
    tf.app.run()
