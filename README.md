# -ONNX-Open-Neural-Network-Exchange

![alt text](https://miro.medium.com/max/948/0*o5doPyWdduatUKtX.PNG)


# About ONNX

- Founded in 2017 by Microsoft .

- ONNX is an intermediary machine learning framework used to convert between different machine learning frameworks

- It is a format designed to represent any type of Machine Learning and Deep Learning model

- Some example of supported frameworks are : PyTorch, TensorFlow, Keras, SAS, Matlab, and many more

- ONNX can make it easier to convert models from one framework to another . Example : PyTorch to TensorFlow , TensorFlow to PyTorch , etc .

- All ONNX models are framework independent .

# Benefits

- Focus on the capabilities needed for inferencing (scoring) , as inferencing speed is very high .

- ONNX is widely supported and can be found in many frameworks, tools, and hardware

- Enabling interoperability between different frameworks and streamlining the path from research to production helps increase the speed of innovation in the AI . 

- File format conversion will be easy using ONNX .

- ONNX makes the file quantized , which saves a lot of space ( Say attributes are of float value format , using .onnx it will convert to int format )        

# Reference Links :


* [Detailed Description](https://github.com/onnx/onnx)

* [ 3 Projects](https://github.com/entbappy/ONNX-Open-Neural-Network-Exchange)

* [ Presentation ](https://docs.google.com/presentation/d/1hqDJgpNmfJul2OkLk8Rry1jGbE9pT-VgKf4_PhojdIw/edit#slide=id.gc6f75fceb_0_0)



# Framework Conversion Example :

- Detectron2 uses pytorch

- If you want to convert from .pt to .pb , then it has to be done via .onnx format .

- Different extension formats are :

		PyTorch : .pt  ( PyTorch easy and PyTorch runtime is faster ) 
		TensorFlow : .pb
		ONNX : .onnx
		model file : .pkl
		h5 file : .h5

# Projects :

### 1 ) ML_Example ( Conversion from ".pkl" -> ".onnx" ) :

- Open the project in VSCode or Pycharm IDE
	
- Setup environment :
	
		conda create -n onnx_ml python==3.7 -y
		conda activate onnx_ml ( source activate onnx_ml )
		pip install -r requirements.txt
	
- create model.pkl :
		python train.py
		
- convert from model.pkl to model.onnx :
		python convert.py
		
		** If any error related to protobuf , run :
		pip install --upgrade protobuf==3.20.0
	
- To predict data run :
	
		python inference.py

### 2 ) ONNX_model_zoo ( mnist Dataset , using onnx pre-trained  )

- Open the project in VSCode or Pycharm IDE
	
- Open terminal :

- You can download the onnx pre-trained model file  "model.onnx" to the folder by : 

		sh download_model.sh
	
- Setup environment :
	
		conda create -n onnx_moldel_zoo python==3.7 -y
		conda activate onnx_moldel_zoo ( or source activate onnx_moldel_zoo )
		pip install -r requirements.txt
		
- To predict data run :
	
		python inference.py	3.png
		
		** If any error related to protobuf , run :
		pip install --upgrade protobuf==3.20.0	

### 3 ) Pytorch_to_Tensoflow ( From ".pb" -> ".onnx" -> ".pt" ) 

- Open "PyTorch to Tensorflow.ipynb" file in google collab .
- Execute all cells one-by-one and observe the output .			


```bash
Author: Bappy Ahmed ( assisted by : Jateendra Pradhan )
Data Scientist
Email: entbappy73@gmail.com

```


