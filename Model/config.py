'''
    @author: chengsx21
    @file: config.py
    @time: 2023/05/01
'''
class Config:
    '''
        This class is used to define the configuration
    '''
    def __init__(self, word2id, word2vec):
        '''
            This function is used to initialize the parameters
        '''
        self.class_num = 2
        self.layer_num = 2
        self.vocab_num = len(word2id) + 1
        self.filter_num = 20
        self.filter_size = [3, 5, 7]
        self.embedding_dim = 50
        self.dropout_prob = 0.3
        self.hidden_size = 100
        self.pretrained_embedding = word2vec
