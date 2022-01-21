import tensorflow as tf
import numpy as np
import cv2
import logging
import os

logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

INPUT_TENSOR_NAME = "input.1:0"
OUTPUT_TENSOR_NAME = "add_4:0"

IMAGE_PATH = "3.png"
PB_PATH = "output/saved_model.pb"

#Preprrocessing the image
img = cv2.imread(IMAGE_PATH)
img = np.dot(img[...,:3], [0.299, 0.587, 0.114]) #convert to grayscale
img = cv2.resize(img, (28, 28), interpolation=cv2.INTER_AREA)
img.resize((1,1,28,28)) #batch of image

with tf.gfile.FastGFile(PB_PATH, 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())

with tf.Graph().as_default() as graph:
    tf.import_graph_def(graph_def, name='')

input_tensor = graph.get_tensor_by_name(INPUT_TENSOR_NAME)
output_tensor = graph.get_tensor_by_name(OUTPUT_TENSOR_NAME)

with tf.Session(graph=graph) as sess:
    output_vals = sess.run(output_tensor, feed_dict={input_tensor: img})

prediction = int(np.argmax(np.array(output_vals).squeeze(), axis=0))
print("Predicted: ",prediction)
