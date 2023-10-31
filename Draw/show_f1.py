'''
    @author: chengsx21
    @file: show_f1.py
    @time: 2023/05/03
'''
import matplotlib.pyplot as plt

train_f1_1, val_f1_1, test_f1_1 = [], [], []
train_f1_2, val_f1_2, test_f1_2 = [], [], []
train_f1_3, val_f1_3, test_f1_3 = [], [], []
train_f1_4, val_f1_4, test_f1_4 = [], [], []

file_list = ["../Log/TextCNN.log", "../Log/LstmRNN.log", "../Log/GruRNN.log", "../Log/MLP.log"]

for i, file in enumerate(file_list):
    with open(file, 'r', encoding='utf-8') as result:
        lines = result.readlines()
        for line in lines:
            if 'Train' in line:
                if i == 0:
                    train_f1_1.append(float(line.split('=')[3].split(',')[0]))
                elif i == 1:
                    train_f1_2.append(float(line.split('=')[3].split(',')[0]))
                elif i == 2:
                    train_f1_3.append(float(line.split('=')[3].split(',')[0]))
                elif i == 3:
                    train_f1_4.append(float(line.split('=')[3].split(',')[0]))
            elif 'Val' in line:
                if i == 0:
                    val_f1_1.append(float(line.split('=')[3].split(',')[0]))
                elif i == 1:
                    val_f1_2.append(float(line.split('=')[3].split(',')[0]))
                elif i == 2:
                    val_f1_3.append(float(line.split('=')[3].split(',')[0]))
                elif i == 3:
                    val_f1_4.append(float(line.split('=')[3].split(',')[0]))
            elif 'Test' in line:
                if i == 0:
                    test_f1_1.append(float(line.split('=')[3].split(',')[0]))
                elif i == 1:
                    test_f1_2.append(float(line.split('=')[3].split(',')[0]))
                elif i == 2:
                    test_f1_3.append(float(line.split('=')[3].split(',')[0]))
                elif i == 3:
                    test_f1_4.append(float(line.split('=')[3].split(',')[0]))

plt.plot(train_f1_1, label='TextCNN')
plt.plot(train_f1_2, label='LstmRNN')
plt.plot(train_f1_3, label='GruRNN')
plt.plot(train_f1_4, label='MLP')
plt.xlabel('Epochs')
plt.title('Train F1_score')
plt.legend()
plt.show()

plt.plot(val_f1_1, label='TextCNN')
plt.plot(val_f1_2, label='LstmRNN')
plt.plot(val_f1_3, label='GruRNN')
plt.plot(val_f1_4, label='MLP')
plt.xlabel('Epochs')
plt.title('Validation F1_score')
plt.legend()
plt.show()

plt.plot(test_f1_1, label='TextCNN')
plt.plot(test_f1_2, label='LstmRNN')
plt.plot(test_f1_3, label='GruRNN')
plt.plot(test_f1_4, label='MLP')
plt.xlabel('Epochs')
plt.title('Test F1_score')
plt.legend()
plt.show()
