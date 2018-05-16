import csv
import cv2
import numpy as np


#importing the driving log
lines = []

with open('./data/driving_log.csv','r') as csvfile:
	reader = csv.reader(csvfile)
	for line in reader:
		lines.append(line)

images = []
measurements = []

#Reading left, right and center images and angle measurements with correction
for line in lines:
	for i in range(3):
		source_path = line[i]
		filename = source_path.split('\\')[-1]
		image = cv2.cvtColor(cv2.imread('./data/IMG/' + filename), cv2.COLOR_BGR2RGB)
		images.append(image)
	correction = 0.2
	measurement = float(line[3])
	measurements.append(measurement)
	measurements.append(measurement + correction)
	measurements.append(measurement - correction)
	
from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Lambda, Convolution2D, Cropping2D, Dropout, Activation
from keras.layers.pooling import MaxPooling2D
from keras.layers import Lambda

#Defining the training set
X_train = np.array(images)
y_train = np.array(measurements)

#Normalising the dataset and cropping the images
model = Sequential()
model.add(Lambda(lambda x: x/255.0 -0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((50,20),(0,0))))

#Using Nvidia model architecture with dropout

model.add(Convolution2D(24, 5, 5, subsample=(2, 2), border_mode="valid")) 
model.add(Activation(activation='relu'))

model.add(Convolution2D(36, 5, 5, subsample=(2, 2), border_mode="valid")) 
model.add(Activation(activation='relu'))

model.add(Convolution2D(48, 5, 5, subsample=(2, 2), border_mode="valid")) 
model.add(Activation(activation='relu'))

model.add(Convolution2D(64, 3, 3, subsample=(2, 2), border_mode="valid")) 
model.add(Activation(activation='relu'))

model.add(Convolution2D(64, 3, 3, subsample=(2, 2), border_mode="valid")) 
model.add(Activation(activation='relu'))

model.add(Dropout(0.2))

model.add(Flatten())

model.add(Dense(100))

model.add(Dense(50))

model.add(Dense(10))

model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, epochs=2)

#Saving the model
model.save('model.h5')