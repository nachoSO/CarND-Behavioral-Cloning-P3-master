import cv2
import numpy as np
import sklearn
from random import shuffle

def process_images(images,measurements):
    augmented_images,augmented_measurements = [],[]
    for image,measurement in zip(images,measurements):
        augmented_images.append(image)
        augmented_measurements.append(measurement)
        augmented_images.append(np.fliplr(image)) #image flipping helps to generalize
        augmented_measurements.append(measurement*-1.0)
    return augmented_images,augmented_measurements

def generator(samples, batch_size=64):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            images = []
            angles = []
            
            for batch_sample in batch_samples:
                name = batch_sample[0].split('/')[-1]
                center_image = cv2.imread(name)
                center_angle = float(batch_sample[3])
                ## steering angle normalization
                ## https://medium.com/@mohankarthik/cloning-a-car-to-mimic-human-driving-5c2f7e8d8aff
                bias=0.5
                threshold = np.random.uniform()
                if (abs(center_angle) + bias) < threshold:
                    pass
                #rgb -> yuv (as is described in the NVIDIA paper)
                img = cv2.cvtColor(center_image, cv2.COLOR_BGR2RGB)
                img = np.asarray(img)
                images.append(img)
                angles.append(center_angle)
                
            # trim image to only see section with road
            augmented_images,augmented_angle=process_images(images,angles)
            X_train = np.array(augmented_images)
            y_train = np.array(augmented_angle)
            yield sklearn.utils.shuffle(X_train, y_train)


