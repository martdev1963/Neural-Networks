import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt 

"""-------------------------------------------------------------------------------------------------------------------------------------------------------------------
								NEURAL NETWORKS PROGRAM
https://www.techwithtim.net/tutorials/python-neural-networks/loading-data/
-------------------------------------------------------------------------------------------------------------------------------------------------------------------"""

# dataset
data = keras.datasets.fashion_mnist

# training and testing dat
(train_images, train_labels), (test_images, test_labels) = data.load_data()

# there are ten labels for this specific dataset... between 0 and 9
# each image that we have will have a specific label assigned to it...

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

#print(train_labels[6])

# by dividing by 255.0, we shrink our data
train_images = train_images/255.0      # the data is in numpy arrays  
test_images = test_images/255.0			

#print(train_images[7])

#plt.imshow(train_images[6], cmap=plt.cm.binary)
#plt.show()

#print(class_names)


# defining the model and layers...in other words; the architecture.
# Whenever you're passing information thats in 2d or 3d array, you need to flatten that 
# information to pass it into an individual neuron as opposed to passing a whole list into a single neuron. 
model = keras.Sequential([ # a sequence of layers
	keras.layers.Flatten(input_shape=(28,28)),	# flattened layer  (INPUT LAYER)
	keras.layers.Dense(128, activation="relu"), # dense layer...a fully connected layer which means, fully connected neural networks (HIDDEN LAYER)
	keras.layers.Dense(10, activation="softmax")  # notice the neuron count of 10....the previous neuron count is 128.    (OUTPUT LAYER)
	#(each neuron is connected to the next layer.)
	]) # relu means: rectified linear unit for the activation function setting.
	# what softmax does is it  picks values for each neuron, so that all of those values add up to one.

# setting up parameters for our model...these are functions that use these algorithms.
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# training our model
model.fit(train_images, train_labels, epochs=5)   #epochs means how many times the model is going to see this information.
# epochs will randomly pick images and labels correspondent to each other, and its going to feed that through the neural network.
# how many epochs you decide is how many times you're going to see the same image.
# the order in which images will come in will influence how parameters and things are tweaked with the network.
# the epochs gives the images in a different order everytime randomly.


# test the model using the test images...to see how many images it gets correct....see if it is learning...
#test_loss, test_acc = model.evaluate(test_images, test_labels)

#print("Tested Acc: ", test_acc)
#print("Tested Loss: ", test_loss)

prediction = model.predict(test_images)
#print(prediction[0]) # prints out a list of lists...
# reflect's the output of the 10 neurons from the OUTPUT LAYER...keras.layers.Dense(10, activation="softmax")

#print(class_names[np.argmax(prediction[0])]) # get's the largest value and finds its index

# forloop to show and see or validate that the model is predicting the images correctly.
for i in range(5):
	plt.grid(False)
	plt.imshow(test_images[i], cmap=plt.cm.binary)
	plt.xlabel("Actual: " + class_names[test_labels[i]])
	plt.title("Prediction " + class_names[np.argmax(prediction[0])])
	plt.show()
