import pickle
import torch
import numpy as np
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F
from .rnn_nn import *
from .base_classifier import *


class RNN_Classifier(Base_Classifier):
    
    def __init__(self,classes=10, input_size=28 , hidden_size=128, activation="relu" ):
        super(RNN_Classifier, self).__init__()

    ############################################################################
    #  TODO: Build a RNN classifier                                            #
    ############################################################################

        self.num_layers = 1
        self.rnn = nn.RNN(input_size, hidden_size, num_layers=self.num_layers)
        layers = [
            nn.Linear(self.num_layers * hidden_size, 2 * self.num_layers * hidden_size),
            nn.PReLU(),
            nn.Linear(2 * self.num_layers * hidden_size, classes),
        ]
        self.classifier_model = nn.Sequential(*layers)

    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################


    def forward(self, x):
    ############################################################################
    #  TODO: Perform the forward pass                                          #
    ############################################################################   

        seq_len, batch_size, _ = x.shape
        _, h = self.rnn(x)
        h = h.transpose(0, 1).contiguous().view(batch_size, -1)
        x = self.classifier_model(h)

    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
        return x


class LSTM_Classifier(Base_Classifier):

    def __init__(self, classes=10, input_size=28, hidden_size=128):
        super(LSTM_Classifier, self).__init__()
        
        #######################################################################
        #  TODO: Build a LSTM classifier                                      #
        #######################################################################

        self.num_layers = 1
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers=self.num_layers)
        layers = [
            nn.Linear(self.num_layers * hidden_size, 2 * self.num_layers * hidden_size),
            nn.PReLU(),
            nn.Linear(2 * self.num_layers * hidden_size, classes),
        ]
        self.classifier_model = nn.Sequential(*layers)

        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################


    def forward(self, x):

        #######################################################################
        #  TODO: Perform the forward pass                                     #
        #######################################################################    

        seq_len, batch_size, _ = x.shape
        _, (h, _) = self.rnn(x)
        h = h.transpose(0, 1).contiguous().view(batch_size, -1)
        x = self.classifier_model(h)

        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
        return x
