import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.models as models
from .Model import Model

import tensorwatch as tw
from keras_sequential_ascii import keras2ascii
from torchsummary import summary

mean = Variable(torch.FloatTensor([0.485, 0.456, 0.406]), requires_grad=False).cuda('cuda:0')
std = Variable(torch.FloatTensor([0.229, 0.224, 0.225]), requires_grad=False).cuda('cuda:0')

def flip(x, dim):
    xsize = x.size()
    dim = x.dim() + dim if dim < 0 else dim
    x = x.view(-1, *xsize[dim:])
    x = x.view(x.size(0), x.size(1), -1)[:, getattr(torch.arange(x.size(1)-1, 
                      -1, -1), ('cpu','cuda:0')[x.is_cuda])().long(), :]
    return x.view(xsize)


class SVCNN(Model):



    def __init__(self, name, nclasses=11, pretraining=True, cnn_name='vgg11'):
        super(SVCNN, self).__init__(name)

        self.classnames=['Alabastron',
        'Amphora',
        'Hydria',
        'Kalathos',
        'Krater',
        'Kylix',
        'Lekythos',
        'Native-American',
        'Pelike',
        'Picher-Shaped',
        'Psykter']

        self.nclasses = nclasses
        self.pretraining = pretraining
        self.cnn_name = cnn_name
        self.use_resnet = cnn_name.startswith('resnet')
        self.mean = Variable(torch.FloatTensor([0.485, 0.456, 0.406]), requires_grad=False).cuda('cuda:0')
        self.std = Variable(torch.FloatTensor([0.229, 0.224, 0.225]), requires_grad=False).cuda('cuda:0')

        if self.use_resnet:
            if self.cnn_name == 'resnet18':
                self.net = models.resnet18(pretrained=self.pretraining)
                self.net.fc = nn.Linear(512,11)
            elif self.cnn_name == 'resnet34':
                self.net = models.resnet34(pretrained=self.pretraining)
                self.net.fc = nn.Linear(512,11)
            elif self.cnn_name == 'resnet50':
                self.net = models.resnet50(pretrained=self.pretraining)
                self.net.fc = nn.Linear(2048,11)
        else:
            if self.cnn_name == 'alexnet':
                self.net_1 = models.alexnet(pretrained=self.pretraining).features
                #model = models.alexnet(pretrained=self.pretraining)
                #model.apply(self.init_weights(model))
                #model_uniform = Net()
                #mynet.apply(self.weights_init_uniform(model))
                #self.net_1 = model.features

                #self.net_1.apply(self.init_weights())
                #self.net_1.apply(self._initialize_weights())
                self.net_2 = models.alexnet(pretrained=self.pretraining).classifier
                #self.net_2 = model.classifier
                #self.net_2 = nn.Linear(512,11)
            elif self.cnn_name == 'vgg11':
                self.net_1 = models.vgg11(pretrained=self.pretraining).features
                self.net_2 = models.vgg11(pretrained=self.pretraining).classifier
            elif self.cnn_name == 'vgg16':
                self.net_1 = models.vgg16(pretrained=self.pretraining).features
                self.net_2 = models.vgg16(pretrained=self.pretraining).classifier
            
            self.net_2._modules['6'] = nn.Linear(4096,11)


    def forward(self, x):
        if self.use_resnet:
            return self.net(x)
        else:
            y = self.net_1(x)
            return self.net_2(y.view(y.shape[0],-1))

    def init_weights(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform(m.weight)
            m.bias.data.fill_(0.)
        elif isinstance(m, nn.Conv2d):
            torch.nn.init.xavier_uniform(m.weight)

class MVCNN(Model):

    def __init__(self, name, model, nclasses=11, cnn_name='vgg11', num_views=12):
        super(MVCNN, self).__init__(name)

        self.classnames=['Alabastron',
        'Amphora',
        'Hydria',
        'Kalathos',
        'Krater',
        'Kylix',
        'Lekythos',
        'Native-American',
        'Pelike',
        'Picher-Shaped',
        'Psykter']
        
        self.nclasses = nclasses
        self.num_views = num_views
        self.mean = Variable(torch.FloatTensor([0.485, 0.456, 0.406]), requires_grad=False).cuda('cuda:0')
        self.std = Variable(torch.FloatTensor([0.229, 0.224, 0.225]), requires_grad=False).cuda('cuda:0')

        self.use_resnet = cnn_name.startswith('resnet')

        if self.use_resnet:
            self.net_1 = nn.Sequential(*list(model.net.children())[:-1])
            self.net_2 = model.net.fc
        else:
            self.net_1 = model.net_1
            self.net_2 = model.net_2
        
        #print(self.net_1)
        #print(self.net_2)

        #alexnet_model = torchvision.models.alexnet()
        #img = tw.model_stats(self.net_1, [1, 3, 224, 224])
        #img.save(r'./alexnet.jpg')
        #keras2ascii(self.net_1)
        #myalexnet = models.alexnet()
        #alexnet.cuda('cuda:0')
        #summary(myalexnet.cuda(), (3, 224, 224))
        #summary(self.net_1, (1,3, 224, 224))


    def forward(self, x):
        #print("X: ",x.shape)
        y = self.net_1(x)
        #print("y: ",y.shape)
        y = y.view((int(x.shape[0]/self.num_views),self.num_views,y.shape[-3],y.shape[-2],y.shape[-1]))#(8,12,512,7,7)
        
        #print("y2: ",y.shape)
        #print("example: ",y[0])
        #print("max: ",torch.max(y,1)[0].view(y.shape[0],-1).shape)
        return self.net_2(torch.max(y,1)[0].view(y.shape[0],-1))

