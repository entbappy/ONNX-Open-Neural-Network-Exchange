from torch.autograd import Variable
import torch
import torch.nn.functional as F
import torch.nn as nn


class Network(nn.Module):
  def __init__(self):
    super(Network, self).__init__()
    self.conv1 = nn.Conv2d(1,20,5,1)
    self.conv2 = nn.Conv2d(20,50,5,1)
    self.fc1 = nn.Linear(4*4*50, 500)
    self.fc2 = nn.Linear(500, 10)

  def forward(self,x):
    x = F.relu(self.conv1(x))
    x = F.max_pool2d(x,2,2)
    x = F.relu(self.conv2(x))
    x = F.max_pool2d(x,2,2)
    x = x.view(-1,4*4*50)
    x = F.relu(self.fc1(x))
    x = self.fc2(x)
    return F.log_softmax(x,dim=1)


trained_model = Network()
trained_model.load_state_dict(torch.load('output/torch_model.pt',map_location ='cpu'))
dummy_input = Variable(torch.randn(1,1,28,28))
torch.onnx.export(trained_model, dummy_input, 'output/onnx_model.onnx')
