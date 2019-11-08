# import the necessary packages
from keras.preprocessing import image as image_utils
from keras.applications.imagenet_utils import decode_predictions
from keras.applications.imagenet_utils import preprocess_input
from keras.applications import VGG16
import numpy as np
import argparse
import cv2
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '2'
# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
    help="path to the input image")
args = vars(ap.parse_args())
 
# load the original image via OpenCV so we can draw on it and display it to our screen later
orig = cv2.imread(args["image"])
print("shape of orig:",orig.shape)
# load the input image using the Keras helper utility while ensuring that the image is resized to 224x224 pxiels, the required input dimensions for the network -- then convert the PIL image to a NumPy array
print("[INFO] loading and preprocessing image...")
'''
image = image_utils.load_img(args["image"], target_size=(224, 224))
image = image_utils.img_to_array(image)
cv2.imshow("downsampled",orig)
cv2.waitKey(3000)
print(image[:,:,0])
print("shape of image:",image.shape)
# our image is now represented by a NumPy array of shape (224, 224, 3), assuming TensorFlow "channels last" ordering of course, but we need to expand the dimensions to be (1, 3, 224, 224) so we can pass it through the network -- we'll also preprocess the image by subtracting the mean RGB pixel intensity from the ImageNet dataset
image = preprocess_input(image)
image = np.expand_dims(image, axis=0)
'''
im = cv2.resize(cv2.imread(args["image"]), (224, 224)).astype(np.float32)
im[:,:,0] -= 103.939
im[:,:,1] -= 116.779
im[:,:,2] -= 123.68
#im = im.transpose((2,0,1))
image = np.expand_dims(im, axis=0)
# load the VGG16 network pre-trained on the ImageNet dataset
print("[INFO] loading network...")
model = VGG16(weights="imagenet")
# classify the image
print("[INFO] classifying image...")
preds = model.predict(image)
print("raw pred array shape:",preds.shape)
P = decode_predictions(preds)
# loop over the predictions and display the rank-5 predictions + probabilities to our terminal
for (i, (imagenetID, label, prob)) in enumerate(P[0]):
    print("{}. {}: {:.2f}%".format(i + 1, label, prob * 100))
# load the image via OpenCV, draw the top prediction on the image, and display the image to our screen
image = cv2.resize(cv2.imread(args["image"]), (224, 224)).astype(np.float32)/255.
cv2.imshow("classification",image)
cv2.waitKey(5000)
