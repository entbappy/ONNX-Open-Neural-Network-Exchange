import onnx
from onnx_tf.backend import prepare

onnx_model = onnx.load('output/onnx_model.onnx')
tf_rep = prepare(onnx_model)
tf_rep.export_graph('output/tf_model.h5')