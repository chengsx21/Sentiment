'''
    @author: chengsx21
    @file: rnn.py
    @time: 2023/04/30
'''
import torch
import torch.nn as nn
from Model.config import Config

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class LstmRNN(nn.Module):
    '''
        This class is used to define the model of LstmRNN
    '''
    def __init__(self, config: Config):
        super(LstmRNN, self).__init__()
        self.class_num = config.class_num
        self.vocab_num = config.vocab_num
        self.layer_num = config.layer_num
        self.hidden_size = config.hidden_size
        self.embedding_dim = config.embedding_dim

        self.embedding = nn.Embedding(self.vocab_num, self.embedding_dim)
        self.embedding.weight.data.copy_(torch.from_numpy(config.pretrained_embedding))
        self.encoder = nn.LSTM(input_size=self.embedding_dim, hidden_size=self.hidden_size, num_layers=self.layer_num, bidirectional=True)
        self.decoder = nn.Linear(2 * self.hidden_size, 64)
        self.fc_layer = nn.Linear(64, self.class_num)

    def forward(self, forward_input):
        '''
            This function is used to define the forward propagation
        '''
        embeddings = self.embedding(forward_input.to(torch.int64)).permute(1, 0, 2)
        _, (h_n, _) = self.encoder(embeddings)
        h_n = h_n.view(self.layer_num, 2, -1, self.hidden_size)
        concat_h_n = torch.cat((h_n[-1, 0], h_n[-1, 1]), dim=-1)
        decoded = self.decoder(concat_h_n)
        return self.fc_layer(decoded)

class GruRNN(nn.Module):
    '''
        This class is used to define the model of GruRNN
    '''
    def __init__(self, config: Config):
        super(GruRNN, self).__init__()
        self.class_num = config.class_num
        self.vocab_num = config.vocab_num
        self.layer_num = config.layer_num
        self.hidden_size = config.hidden_size
        self.embedding_dim = config.embedding_dim

        self.embedding = nn.Embedding(self.vocab_num, self.embedding_dim)
        self.embedding.weight.data.copy_(torch.from_numpy(config.pretrained_embedding))
        self.encoder = nn.GRU(input_size=self.embedding_dim, hidden_size=self.hidden_size, num_layers=self.layer_num, bidirectional=True)
        self.decoder = nn.Linear(2 * self.hidden_size, 64)
        self.fc_layer = nn.Linear(64, self.class_num)

    def forward(self, forward_input):
        '''
            This function is used to define the forward propagation
        '''
        embeddings = self.embedding(forward_input.to(torch.int64)).permute(1, 0, 2)
        h_0 = torch.rand(self.layer_num * 2, embeddings.size(1), self.hidden_size).to(device=device)
        _, h_n = self.encoder(embeddings, h_0)
        h_n = h_n.view(self.layer_num, 2, -1, self.hidden_size)
        concat_h_n = torch.cat((h_n[-1, 0], h_n[-1, 1]), dim=-1)
        decoded = self.decoder(concat_h_n)
        return self.fc_layer(decoded)

if __name__ == '__main__':
    print('This is the model of RNN')
