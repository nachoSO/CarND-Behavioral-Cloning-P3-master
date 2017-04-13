import cv2
import numpy as np
import sklearn
from random import shuffle

def process_images(images,measurements):
    augmented_images,augmented_measurements = [],[]
    for image,measurement in zip(images,measurements):
        augmented_images.append(image)
        augmented_measurements.append(measurement)
        augmented_images.append(cv2.flip(image,1)) #image flipping helps to generalize
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
                #bias control
                bias=0.5
                threshold = np.random.uniform()
                if (abs(center_angle) + bias) < threshold:
                    pass
                change_brightness(center_image) #change lightness
                images.append(center_image)
                angles.append(center_angle)
                
            # trim image to only see section with road
            augmented_images,augmented_angle=process_images(images,angles)
            X_train = np.array(augmented_images)
            y_train = np.array(augmented_angle)
            yield sklearn.utils.shuffle(X_train, y_train)

# "Changing brightness allows the model to become robust towards all lighting conditions."
# https://medium.com/@mohankarthik/cloning-a-car-to-mimic-human-driving-5c2f7e8d8aff
def change_brightness(img):
    temp = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Compute a random brightness value and apply to the image
    brightness = 0.3 + np.random.uniform()
    temp[:, :, 2] = temp[:, :, 2] * brightness

    # Convert back to RGB and return
    return cv2.cvtColor(temp, cv2.COLOR_HSV2RGB)

