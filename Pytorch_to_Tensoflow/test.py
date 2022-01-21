import tensorflow as tf
import cv2
import numpy as np


IMAGE_PATH = "3.png"
#Preprrocessing the image
img = cv2.imread(IMAGE_PATH)
img = np.dot(img[...,:3], [0.299, 0.587, 0.114]) #convert to grayscale
img = cv2.resize(img, (28, 28), interpolation=cv2.INTER_AREA)
img.resize((1,1,28,28)) #batch of image

new_model = tf.keras.models.load_model('output/tf_model/')
new_model.predict(img)
