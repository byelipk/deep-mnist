import cv2
import numpy as np
import argparse
import matplotlib.pyplot as plt
from keras.datasets import mnist

ap = argparse.ArgumentParser()
ap.add_argument(
    "-t", "--target", required=True, help="Target label of image.")
ap.add_argument(
    "-i", "--image", required=True, help="Image to classify.")
ap.add_argument(
    "-r", "--run", help="Run the model on IMAGE.")
ap.add_argument(
    "-s", "--show", help="Show preprocessing steps on IMAGE.")
ap.add_argument(
    "-c", "--compare", help="Compare preprocessed IMAGE with test image.")

args = vars(ap.parse_args())

# Which digit are we trying to predict?
TARGET_LABEL = int(args["target"])
IMAGE_PATH = args["image"]

# Cropping dimensions
x_start = 100
x_end   = 1300
y_start = 200
y_end   = 900

# Load and preprocess test image
image   = cv2.imread(IMAGE_PATH)
cropped = image[x_start:x_end, y_start:y_end]
gray    = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5,5), 0)


threshold = 165
maximum   = 255 # Any pixel intensity > threshold is set to this value
dims      = (28, 28) # All mnist images this 28 x 28
(T, threshInv) = cv2.threshold(blurred,
                               threshold,
                               maximum,
                               cv2.THRESH_BINARY_INV)

resized = cv2.resize(threshInv, dims, interpolation=cv2.INTER_AREA)


if args["show"]:
    # Visualize different versions of the image:
    cv2.imshow("Cropped", cropped)
    cv2.imshow("Gray", gray)
    cv2.imshow("Blurred", blurred)
    cv2.imshow("Threshold", threshInv)
    cv2.imshow("Resized", resized)
    cv2.waitKey(0)



if args["compare"]:
    # Visualize our image and another 5 in the mnist test set:
    target_img = None
    (_, _), (test_images, test_labels) = mnist.load_data()
    for index, label in enumerate(test_labels):
        if label == TARGET_LABEL:
            target_img = test_images[index]
            break

    # Draw the plot:
    plt.subplot(330 + 1 + 0)
    plt.imshow(target_img, cmap=plt.get_cmap("gray"))
    plt.subplot(330 + 1 + 1)
    plt.imshow(resized, cmap=plt.get_cmap("gray"))
    plt.show()


if args["run"]:
    from keras.models import load_model
    from keras.utils.np_utils import to_categorical

    # Load our trained model
    network = load_model("mnist_model.h5")

    # Preprocess image and label
    test_img   = resized
    test_img   = test_img.reshape((1, 28 * 28))
    test_img   = test_img.astype("float32") / 255
    test_label = to_categorical(np.array(TARGET_LABEL), 10)

    # Does our classifier know the image is a 5?
    test_loss, test_acc = network.evaluate(test_img, test_label)
    print()
    print("Test accuracy:", test_acc)
    print("Test loss:", test_loss)
