'''
    @author: chengsx21
    @file: main.py
    @time: 2023/04/29
'''
import argparse
import logging
import gensim
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from sklearn import metrics
from Model import cnn, rnn, mlp, config

def get_word2vec():
    '''
        This function is used to get the word2id and word2vec
    '''
    file_list = ["./Dataset/train.txt", "./Dataset/validation.txt"]
    pretrained_model = "./Dataset/wiki_word2vec_50.bin"
    vec_model = gensim.models.KeyedVectors.load_word2vec_format(pretrained_model, binary=True)
    word_to_id = {}
    word_num = 0
    for file in file_list:
        with open(file, "r", encoding="utf-8") as read_file:
            for line in read_file.readlines():
                for word in line.strip().split()[1:]:
                    if word not in word_to_id:
                        word_to_id[word] = word_num
                        word_num += 1
    word_to_vec = np.array(np.zeros([len(word_to_id) + 1, vec_model.vector_size]))
    for key, _ in word_to_id.items():
        try:
            word_to_vec[word_to_id[key]] = vec_model[key]
        except Exception:
            pass
    return word_to_id, word_to_vec

word2id, word2vec = get_word2vec()
config = config.Config(word2id=word2id, word2vec=word2vec)

def pre_processing(file: str, word_to_id: dict, length=120):
    '''
        This function is used to get the corpus
    '''
    contents, labels = np.array([0] * length), np.array([])
    with open(file, encoding="utf-8") as read_file:
        for line in read_file.readlines():
            sentence = line.strip().split()
            attitudes, comments = int(sentence[0]), sentence[1:]
            content = np.asarray([word_to_id.get(word, 0) for word in comments])[:length]
            content = np.pad(content, ((0, max(length - len(content), 0))), "constant", constant_values=0)
            labels = np.append(labels, attitudes)
            contents = np.vstack([contents, content])
    contents = np.delete(contents, 0, axis=0)
    return contents, labels

def get_dataloader(file: str):
    '''
        This function is used to get the dataloader
    '''
    contents, labels = pre_processing(file, word2id)
    dataset = TensorDataset(torch.from_numpy(contents).type(torch.float),torch.from_numpy(labels).type(torch.long))
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, num_workers=2, shuffle=True)
    return dataloader

def get_train_dataloader():
    '''
        This function is used to get the train dataloader
    '''
    return get_dataloader("./Dataset/train.txt"), get_dataloader("./Dataset/validation.txt"), get_dataloader("./Dataset/test.txt")

def get_parameters():
    '''
        This function is used to get the parameters
    '''
    model_optimizer = torch.optim.Adam(model.parameters(), lr=learn_rate)
    return model_optimizer, nn.CrossEntropyLoss(), torch.optim.lr_scheduler.StepLR(model_optimizer, step_size=5)

def get_command():
    '''
        This function is used to parse the arguments
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument("-l",  dest="learn_rate",        type=float,  default=1e-3)
    parser.add_argument("-e",  dest="epoch",             type=int,    default=10)
    parser.add_argument("-b",  dest="batch_size",        type=int,    default=50)
    parser.add_argument("-p",  dest="pretrained_model",  type=str,    default="CNN",)
    args = parser.parse_args()
    if args.pretrained_model == "CNN":
        network = cnn.TextCNN(config=config).to(device)
    elif args.pretrained_model == "RNN1":
        network = rnn.LstmRNN(config=config).to(device)
    elif args.pretrained_model == "RNN2":
        network = rnn.GruRNN(config=config).to(device)
    elif args.pretrained_model == "MLP":
        network = mlp.MLP(config=config).to(device)
    else:
        print("INVALID NEURAL NETWORK")
        exit(-1)
    return args.learn_rate, args.epoch, args.batch_size, network

def update_model(input_data, target, output, loss, model_loss, correct, count, full_true, full_pred):
    '''
        This function is used to update the parameters
    '''
    model_loss += loss.item()
    correct += (output.argmax(1) == target).float().sum().item()
    count += len(input_data)
    full_true.extend(target.cpu().numpy().tolist())
    full_pred.extend(output.argmax(1).cpu().numpy().tolist())
    return model_loss, correct, count, full_true, full_pred

def get_result_of_model(dataloader, model_batch_size, model_loss, correct, count, full_true, full_pred):
    '''
        This function is used to get the result
    '''
    model_acc, model_f1 = 0.0, 0.0
    model_loss *= model_batch_size
    model_loss /= len(dataloader.dataset)
    model_acc = correct / count
    model_f1 = metrics.f1_score(np.array(full_true), np.array(full_pred), average="binary")
    return model_loss, model_acc, model_f1

def train_model(dataloader, network_model: nn.Module):
    '''
        This function is used to train the model
    '''
    network_model.train()
    train_loss, train_acc, count, correct, full_true, full_pred = 0.0, 0.0, 0, 0, [], []
    for _, (input_data, target) in enumerate(dataloader):
        input_data, target = input_data.to(device), target.to(device)
        optimizer.zero_grad()
        output = network_model(input_data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        train_loss, correct, count, full_true, full_pred = update_model(input_data, target, output, loss, train_loss, correct, count, full_true, full_pred)
    scheduler.step()
    train_loss, train_acc, train_f1 = get_result_of_model(dataloader, batch_size, train_loss, correct, count, full_true, full_pred)
    return train_loss, train_acc, train_f1

def evaluate_model(dataloader, network_model: nn.Module):
    '''
        This function is used to evaluate the model
    '''
    network_model.eval()
    eval_loss, eval_acc, count, correct, full_true, full_pred = 0.0, 0.0, 0, 0, [], []
    for _, (input_data, target) in enumerate(dataloader):
        input_data, target = input_data.to(device), target.to(device)
        output = network_model(input_data)
        loss = criterion(output, target)
        eval_loss, correct, count, full_true, full_pred = update_model(input_data, target, output, loss, eval_loss, correct, count, full_true, full_pred)
    eval_loss, eval_acc, eval_f1 = get_result_of_model(dataloader, batch_size, eval_loss, correct, count, full_true, full_pred)
    return eval_loss, eval_acc, eval_f1

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    learn_rate, epoch, batch_size, model = get_command()
    train_dataloader, val_dataloader, test_dataloader = get_train_dataloader()
    optimizer, criterion, scheduler = get_parameters()

    logger = logging.getLogger(__name__)
    logging.basicConfig(filename=f'{model.__class__.__name__}.log', level=logging.INFO)

    for cur_epoch in tqdm(range(1, epoch + 1)):
        print("\033[1mRunning epoch:\033[0m " , cur_epoch, "    \033[1mModel\033[0m: ", model.__class__.__name__)
        tr_loss, tr_acc, tr_f1 = train_model(train_dataloader, model)
        va_loss, va_acc, va_f1 = evaluate_model(val_dataloader, model)
        te_loss, te_acc, te_f1 = evaluate_model(test_dataloader, model)
        logger.info("Epoch %d: Train_loss=%.4f, Train_acc=%.4f, Train_f1=%.4f", cur_epoch, tr_loss, tr_acc, tr_f1)
        logger.info("Epoch %d: Val_loss=%.4f, Val_acc=%.4f, Val_f1=%.4f", cur_epoch, va_loss, va_acc, va_f1)
        logger.info("Epoch %d: Test_loss=%.4f, Test_acc=%.4f, Test_f1=%.4f", cur_epoch, te_loss, te_acc, te_f1)
        print(f"\033[1m\033[31mTrain_loss:\033[0m {tr_loss:.4f},    \033[1m\033[31mTrain_acc:\033[0m {tr_acc:.4f},    \033[1m\033[31mTrain_f1:\033[0m {tr_f1:.4f}")
        print(f"\033[1m\033[33mVal_loss:\033[0m   {va_loss:.4f},    \033[1m\033[33mVal_acc:\033[0m   {va_acc:.4f},    \033[1m\033[33mVal_f1:\033[0m   {va_f1:.4f}")
        print(f"\033[1m\033[34mTest_loss:\033[0m  {te_loss:.4f},    \033[1m\033[34mTest_acc:\033[0m  {te_acc:.4f},    \033[1m\033[34mTest_f1:\033[0m  {te_f1:.4f}")
