import onnx
from onnx_tf.backend import prepare
import numpy as np
from PIL import Image

# Load the ONNX file
model = onnx.load('output/onnx_model.onnx')

# Import the ONNX model to Tensorflow
tf_rep = prepare(model)



print('Image 1:')
img = Image.open('3.png').resize((28, 28)).convert('L')

output = tf_rep.run(np.asarray(img, dtype=np.float32)[np.newaxis, np.newaxis, :, :])
print('The digit is classified as ', np.argmax(output))
print('------------------------------------------------------------------------------')
print('Image 2:')
img = Image.open('7.png').resize((28, 28)).convert('L')

output = tf_rep.run(np.asarray(img, dtype=np.float32)[np.newaxis, np.newaxis, :, :])
print('The digit is classified as ', np.argmax(output))

tf_rep.export_graph('mnist.pb')