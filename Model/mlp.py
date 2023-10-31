'''
    @author: chengsx21
    @file: mlp.py
    @time: 2023/04/30
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from Model.config import Config

class MLP(nn.Module):
    '''
        This class is used to define the model of MLP
    '''
    def __init__(self, config: Config):
        super(MLP, self).__init__()
        self.class_num = config.class_num
        self.vocab_num = config.vocab_num
        self.layer_num = config.layer_num
        self.hidden_size = config.hidden_size
        self.embedding_dim = config.embedding_dim

        self.embedding = nn.Embedding(self.vocab_num, self.embedding_dim)
        self.embedding.weight.data.copy_(torch.from_numpy(config.pretrained_embedding))
        self.mlp_layer = torch.nn.Linear(self.embedding_dim, self.hidden_size)
        self.linear = torch.nn.Linear(self.hidden_size, self.class_num)
        self.relu = torch.nn.ReLU()

        for _, parameter in self.named_parameters():
            if parameter.requires_grad:
                torch.nn.init.normal_(parameter, mean = 0, std = 0.01)

    def forward(self, forward_input):
        '''
            This function is used to define the forward propagation
        '''
        embeddings = self.embedding(forward_input.to(torch.int64))
        mlp_output = self.relu(self.mlp_layer(embeddings))
        max_pooled = F.max_pool1d(mlp_output.permute(0, 2, 1), mlp_output.size(2)).squeeze(2)
        return self.linear(max_pooled)

if __name__ == '__main__':
    print('This is the model of MLP')
