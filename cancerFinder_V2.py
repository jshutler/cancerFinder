from PIL import Image
import numpy as np
import tensorflow as tf 
import pandas as pd 
import tensorflow.keras as keras
import matplotlib.pyplot as plt 
import glob
import os
import ntpath
import h5py



def main():
	#imArrays is a list of Arrays with dimensions 96x96x3
	imArrays, imLabels, trainingDataFrame, testingArrays, testingDataFrame = loadData()
	print(trainingDataFrame)
	print(testingDataFrame)
	# visualizeData(imArrays, imLabels)
	learning(imArrays, imLabels, testingArrays, testingDataFrame)



def loadData():
	'''Base Load Data function. Allows you to select if you want both the training and testing Data loaded, or just one or the other.'''

	imArrays, imLabels, trainingDataFrame = loadTraining()

	flag = True
	

	#if we want to load all the Testing Data. We'll switch flag to True
	if flag == True:
		testingArrays, testingDataFrame = loadTesting()
		
	#imArrays are the training arrays, 
	return imArrays, imLabels, trainingDataFrame, testingArrays, testingDataFrame
	
def loadTraining():
	trainingDataFrame = pd.read_csv('cancer_images/train_labels.csv')
	imArrays = []
	#There are 220,024 images in the training set 
	numDataPoints = 100

	#downloads the first 1000 images from the folder
	for i in range(numDataPoints):
		im = Image.open('cancer_images/train/' + str(trainingDataFrame['id'][i]) + '.tif')
		imArrays.append(im)
		im.load()

	#turns all the images into arrays
	for i in range(len(imArrays)):
		imArrays[i] = np.array(imArrays[i])

	
	#NOTE: IF THIS FAILS, TRY CONVERTING TO VECTORIZED BINARY VALUES
	imLabels = np.array(trainingDataFrame["label"][:numDataPoints])
	imArrays = np.array(imArrays)

	print(imArrays.shape)
	


	return imArrays, imLabels, trainingDataFrame

def loadTesting():
	#blank list for all the test ing Arrays
	testingArrays = []
	#Blank list for getting all the Testing ID's
	IDs = []
	#puts all the testing ID's i na dictionary so it will be easier to put into a DataFrame later
	testingIDs = {"ID": IDs}

	numTestingImages = 100

	#grabs the path to each image in the data set
	f = glob.glob(os.path.join("cancer_images/test", "*.tif"))



	#Takes all the Image paths and gets both the image array, and the image tag
	for i in range(numTestingImages):
		#opens image
		im = Image.open(f[i])
		#appends the image tag to the list, so it gets added to the dictionary
		IDs.append(ntpath.basename(f[i][:-4]))
		#appends the testing arrays to the list
		testingArrays.append(np.array(im))
		#closes image file to save memory
		im.load()

	testingDataFrame = pd.DataFrame({"IDs": IDs})

		
	testingArrays = np.array(testingArrays)



	


	return testingArrays, testingDataFrame

def visualizeData(imArrays, imLabels):

	# import sys
	# np.set_printoptions(threshold=sys.maxsize)
	# plt.imshow(imArrays[1][:,:,0])
	plt.imshow(imArrays[1])
	plt.show()

def learning(imArrays, imLabels, testingArrays, testingDataFrame):
	'''Where the actual Machine Learning takes place'''


	model = keras.models.Sequential()

	#where I built the layers of the object.
	model.add(keras.layers.Conv2D(64, kernel_size=3, activation="relu", input_shape=(96,96,3)))
	model.add(keras.layers.AveragePooling2D())
	model.add(keras.layers.Conv2D(32, kernel_size=3, activation="relu"))
	model.add(keras.layers.AveragePooling2D())
	model.add(keras.layers.Flatten())
	model.add(keras.layers.Dense(32, activation="relu"))
	model.add(keras.layers.Dense(32, activation="relu"))
	#sigmoid acitivation
	model.add(keras.layers.Dense(1, activation="sigmoid"))

	model.summary()

	model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(lr=0.0001, decay=0.00001), metrics=["accuracy"])

	history = model.fit(imArrays, imLabels, batch_size=32, epochs=10, verbose=2)

	prediction = model.predict(testingArrays)

	# model_json = model.to_json()
	# with open("model.json", "w") as json_file:
	# 	json_file.write(model_json)

	# model.to_weights('cancer_model.5h')


	visualizeModel(model, history)

	testingDataFrame['predictions'] = prediction

	testingDataFrame.to_csv('predictions.csv')


def visualizeModel(model, history):
	# Plot training & validation accuracy values
	plt.plot(history.history['acc'])
	# plt.plot(history.history['val_acc'])
	plt.title('Model accuracy')
	plt.ylabel('Accuracy')
	plt.xlabel('Epoch')
	plt.legend(['Train', 'Test'], loc='upper left')
	plt.show()

	# Plot training & validation loss values
	plt.plot(history.history['loss'])
	# plt.plot(history.history['val_loss'])
	plt.title('Model loss')
	plt.ylabel('Loss')
	plt.xlabel('Epoch')
	plt.legend(['Train', 'Test'], loc='upper left')
	plt.show()




def showImage(vecImage):
	'''Takes the vector of an image, and shows the original image'''
	plt.imshow(vecImage)
	plt.show()
if __name__ == '__main__':
	main()