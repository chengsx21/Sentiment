'''
    @author: chengsx21
    @file: cnn.py
    @time: 2023/04/30
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from Model.config import Config

class TextCNN(nn.Module):
    '''
        This class is used to define the model of TextCNN
    '''
    def __init__(self, config: Config):
        '''
            This function is used to initialize the model
        '''
        super(TextCNN, self).__init__()
        self.class_num = config.class_num
        self.vocab_num = config.vocab_num
        self.embedding_dim = config.embedding_dim
        self.dropout_prob = config.dropout_prob
        self.filter_num = config.filter_num
        self.filter_size = config.filter_size

        self.embedding = nn.Embedding(self.vocab_num, self.embedding_dim)
        self.embedding.weight.data.copy_(torch.from_numpy(config.pretrained_embedding))
        self.convs_layer = nn.ModuleList([nn.Conv2d(1, self.filter_num, (k, self.embedding_dim)) for k in self.filter_size])
        self.dropout = nn.Dropout(self.dropout_prob)
        self.fc_layer = nn.Linear(self.filter_num * len(self.filter_size), self.class_num)

    def conv_and_pool(self, embedded_input, conv_layer):
        '''
            This function is used to define the convolution and pooling layer
        '''
        conv_output = F.relu(conv_layer(embedded_input).squeeze(3))
        conv_pooled_features = F.max_pool1d(conv_output, conv_output.size(2)).squeeze(2)
        return conv_pooled_features

    def forward(self, forward_input):
        '''
            This function is used to define the forward propagation
        '''
        embedded_input = self.embedding(forward_input.to(torch.int64)).unsqueeze(1)
        conv_pooled_features = [self.conv_and_pool(embedded_input, conv_layer) for conv_layer in self.convs_layer]
        return (F.log_softmax(self.fc_layer(self.dropout(torch.cat((conv_pooled_features), 1))), dim=1))

if __name__ == '__main__':
    print('This is the model of CNN')
