import numpy as np
from keras.models import load_model

test_images = np.load("mnist_test_images.npy")
test_labels = np.load("mnist_test_labels.npy")

network = load_model("mnist_model.h5")

# Evaluate the model
print("Evaluating...")
test_loss, test_acc = network.evaluate(test_images, test_labels)
print()
print("Test accuracy:", test_acc)
print("Test loss:", test_loss)
