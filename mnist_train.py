from keras.datasets import mnist
import matplotlib.pyplot as plt
import numpy as np

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Display images in a 3x3 grid
for i in range(0,9):
    plt.subplot(330 + 1 + i)
    plt.imshow(train_images[i], cmap=plt.get_cmap("gray"))
plt.show()

# The network
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout

network = Sequential()
network.add(Dense(512, activation="relu", input_shape=(28 * 28,)))
network.add(Dense(10, activation="softmax"))

network.compile(optimizer="rmsprop",
                loss="categorical_crossentropy",
                metrics=['accuracy'])


# NOTE
# Before training, we will preprocess our data by reshaping it how
# the network expects, and scaling it so that all values are in the [0, 1]
# interval. Previously, our training images for instance were stored in an
# array of shape (60000, 28, 28) of type uint8 with values in the [0, 255]
# interval. We transform it into a float32 array of shape (60000, 28 * 28) with
# values between 0 and 1.
train_images = train_images.reshape((60000, 28 * 28))
train_images = train_images.astype('float32') / 255

test_images = test_images.reshape((10000, 28 * 28))
test_images = test_images.astype('float32') / 255

# One-hot-encode the labels
from keras.utils.np_utils import to_categorical

train_labels = to_categorical(train_labels)
test_labels  = to_categorical(test_labels)

# Save test data
np.save("mnist_test_images", test_images)
np.save("mnist_test_labels", test_labels)

# Train the model
network.fit(train_images, train_labels, nb_epoch=5, batch_size=128)

network.save("mnist_model.h5")
