from keras.applications.vgg16 import VGG16,decode_predictions
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
import numpy as np
import os
import copy

def vgg_preprocess(im):
  x = copy.deepcopy(im*255.)
  x = np.expand_dims(x, axis=0)
  x = preprocess_input(x)
  return x

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
filename = 'images/beer.png'
model = VGG16(weights='imagenet')
#img = image.load_img(filename, target_size=(224, 224))
#x = image.img_to_array(img)
im = image.img_to_array(image.load_img(filename, target_size=(224, 224)))/255.
#x = np.expand_dims(x, axis=0)
#x = preprocess_input(x)
x = vgg_preprocess(im)

features = model.predict(x)
P = decode_predictions(features)
# loop over the predictions and display the rank-5 predictions + probabilities to our terminal
for (i, (imagenetID, label, prob)) in enumerate(P[0]):
    print("{}. {}: {:.2f}%".format(i + 1, label, prob * 100))
print("range of feature array")
print(features.min(),features.max())
