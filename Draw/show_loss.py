'''
    @author: chengsx21
    @file: show_loss.py
    @time: 2023/05/03
'''
import matplotlib.pyplot as plt

train_loss_1, val_loss_1, test_loss_1 = [], [], []
train_loss_2, val_loss_2, test_loss_2 = [], [], []
train_loss_3, val_loss_3, test_loss_3 = [], [], []
train_loss_4, val_loss_4, test_loss_4 = [], [], []

file_list = ["../Log/TextCNN.log", "../Log/LstmRNN.log", "../Log/GruRNN.log", "../Log/MLP.log"]

for i, file in enumerate(file_list):
    with open(file, 'r', encoding='utf-8') as result:
        lines = result.readlines()
        for line in lines:
            if 'Train' in line:
                if i == 0:
                    train_loss_1.append(float(line.split('=')[1].split(',')[0]))
                elif i == 1:
                    train_loss_2.append(float(line.split('=')[1].split(',')[0]))
                elif i == 2:
                    train_loss_3.append(float(line.split('=')[1].split(',')[0]))
                elif i == 3:
                    train_loss_4.append(float(line.split('=')[1].split(',')[0]))
            elif 'Val' in line:
                if i == 0:
                    val_loss_1.append(float(line.split('=')[1].split(',')[0]))
                elif i == 1:
                    val_loss_2.append(float(line.split('=')[1].split(',')[0]))
                elif i == 2:
                    val_loss_3.append(float(line.split('=')[1].split(',')[0]))
                elif i == 3:
                    val_loss_4.append(float(line.split('=')[1].split(',')[0]))
            elif 'Test' in line:
                if i == 0:
                    test_loss_1.append(float(line.split('=')[1].split(',')[0]))
                elif i == 1:
                    test_loss_2.append(float(line.split('=')[1].split(',')[0]))
                elif i == 2:
                    test_loss_3.append(float(line.split('=')[1].split(',')[0]))
                elif i == 3:
                    test_loss_4.append(float(line.split('=')[1].split(',')[0]))

plt.plot(train_loss_1, label='TextCNN')
plt.plot(train_loss_2, label='LstmRNN')
plt.plot(train_loss_3, label='GruRNN')
plt.plot(train_loss_4, label='MLP')
plt.xlabel('Epochs')
plt.title('Train Loss')
plt.legend()
plt.show()

plt.plot(val_loss_1, label='TextCNN')
plt.plot(val_loss_2, label='LstmRNN')
plt.plot(val_loss_3, label='GruRNN')
plt.plot(val_loss_4, label='MLP')
plt.xlabel('Epochs')
plt.title('Validation Loss')
plt.legend()
plt.show()

plt.plot(test_loss_1, label='TextCNN')
plt.plot(test_loss_2, label='LstmRNN')
plt.plot(test_loss_3, label='GruRNN')
plt.plot(test_loss_4, label='MLP')
plt.xlabel('Epochs')
plt.title('Test Loss')
plt.legend()
plt.show()
