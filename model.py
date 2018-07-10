import tensorflow as tf

from keras.models import Sequential
from keras.layers import Lambda,Cropping2D,Activation,Flatten,Dense,Dropout,ELU
from keras.layers.convolutional import Convolution2D
from keras.optimizers import Adam

from sklearn.model_selection import train_test_split # to split training data into train and validation data

import numpy as np 
import csv
import cv2




def create_model():
	#create a model wrt Nvidia end to end autonomous vehicle architecture 
	model = Sequential()
	#add normalization layer
	model.add(Lambda(lambda x : (x/255)-0.5, input_shape = (160,320,3)))
	#add cropping layer
	model.add(Cropping2D(cropping = ((70,25),(0,0))))
	#add convolution layer with filter size as 24, patch size of 5*5 and stride of 2*2 with relu activation
	model.add(Convolution2D(24, 5, 5, subsample = (2,2), activation = 'relu'))
	#add convolution layer with filter size as 36, patch size of 5*5 and stride of 2*2 with relu activation
	model.add(Convolution2D(36, 5, 5, subsample = (2,2), activation = 'relu'))
	#add convolution layer with filter size as 48, patch size of 5*5 and stride of 2*2 with relu activation
	model.add(Convolution2D(48, 5, 5, subsample = (2,2), activation = 'relu'))
	#add convolution layer with filter size as 64, patch size of 3*3 and stride of 1*1 with relu activation
	model.add(Convolution2D(64, 3, 3, subsample = (1,1), activation = 'relu'))
	#add convolution layer with filter size as 64, patch size of 3*3 and stride of 1*1 with relu activation
	model.add(Convolution2D(64, 3, 3, subsample = (1,1), activation = 'relu'))
	#flatten the output from thee previous layer to pass to fully connected layer
	model.add(Flatten())
	#add fully connected layer with 100 neurons
	model.add(Dense(100))
	#add fully connected layer with 50 neurons
	model.add(Dense(50))
	#add fully conncected layer with 10 neurons
	model.add(Dense(10))
	#add output layer with one neuron as we are dealing with a regression problem
	model.add(Dense(1))
	return model
 
#function to retrieve training data
def get_training_data():
	csv_file = 'driving_log.csv'
	path = 'data/'
	csv_path = path + csv_file
	images = []
	steering_angles = []
	correction = 0.2
	with open(csv_path) as data_file:
		reader = csv.reader(data_file)
		for line in reader:
			measurement = float(line[3])
			temp_img_path = line[0].split("\\")
			final_img_path = temp_img_path[len(temp_img_path)-2]+ '/' + temp_img_path[len(temp_img_path)-1]
			center_img_path = path + final_img_path
			#print(center_img_path)
			center_image = cv2.imread(center_img_path)
			center_image_rgb = cv2.cvtColor(center_image, cv2.COLOR_BGR2RGB)
			images.append(center_image_rgb)
			steering_angles.append(measurement)
			images.append(cv2.flip(center_image_rgb,1))
			steering_angles.append(-1*measurement)
			temp_img_path = line[1].split("\\")
			final_img_path = temp_img_path[len(temp_img_path)-2]+ '/' + temp_img_path[len(temp_img_path)-1]
			left_img_path = path + final_img_path
			left_image = cv2.imread(left_img_path)
			left_image_rgb = cv2.cvtColor(left_image, cv2.COLOR_BGR2RGB)
			images.append(left_image_rgb)
			steering_angles.append(measurement + correction)
			images.append(cv2.flip(left_image_rgb,1))
			steering_angles.append(-1*(measurement + correction))
			temp_img_path = line[2].split("\\")
			final_img_path = temp_img_path[len(temp_img_path)-2]+ '/' + temp_img_path[len(temp_img_path)-1]
			right_img_path = path + final_img_path
			right_image = cv2.imread(right_img_path)
			right_image_rgb = cv2.cvtColor(right_image, cv2.COLOR_BGR2RGB)
			images.append(right_image_rgb)
			steering_angles.append(measurement - correction)
			images.append(cv2.flip(right_image_rgb,1))
			steering_angles.append(-1*(measurement - correction))
	images = np.array(images)
	steering_angles = np.array(steering_angles)
	return images,steering_angles

def generator(samples, batch_size = 32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                name = './IMG/'+batch_sample[0].split('/')[-1]
                center_image = cv2.imread(name)
                center_angle = float(batch_sample[3])
                images.append(center_image)
                angles.append(center_angle)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)


X_train, Y_train = get_training_data()
if(len(X_train) == len(Y_train)):
        print("size is : " + str(len(X_train)) )
else:
        print("something wrong")

X_train, X_valid, Y_train, Y_valid = train_test_split(X_train,Y_train, test_size = 0.2) 
learning_rate = 0.0001
adam = Adam(lr = learning_rate)
model = create_model()
model.summary()
model.compile(loss = 'mse', optimizer = adam)
model.fit(X_train, Y_train, nb_epoch=12, verbose = 1, batch_size = 128, shuffle = True, validation_data = (X_valid, Y_valid))
model.save('model.h5')
