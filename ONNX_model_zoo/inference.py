import sys
import json
import cv2
import numpy as np
import onnx
import onnxruntime
from onnx import numpy_helper

model_dir = './mnist' # model directory
model = model_dir + '/model.onnx' # model file
path = sys.argv[1] # image path

#Preprrocessing the image
img = cv2.imread(path)
img = np.dot(img[...,:3], [0.299, 0.587, 0.114]) #convert to grayscale
img = cv2.resize(img, (28, 28), interpolation=cv2.INTER_AREA)
img.resize((1,1,28,28)) #batch of image

data = json.dumps({"data": img.tolist()})
data = np.array(json.loads(data)["data"]).astype(np.float32)
session = onnxruntime.InferenceSession(model, None)
input_name = session.get_inputs()[0].name
ouput_name = session.get_outputs()[0].name

result = session.run([ouput_name], {input_name: data})
prediction = int(np.argmax(np.array(result).squeeze(), axis=0))
print("Predicted: ",prediction)