'''
    @author: chengsx21
    @file: show_lr.py
    @time: 2023/05/03
'''
import matplotlib.pyplot as plt

train_acc_1, train_f1_1 = [], []
train_acc_2, train_f1_2 = [], []
train_acc_3, train_f1_3 = [], []
train_acc_4, train_f1_4 = [], []

file = "../Log/LearningRate.log"
with open(file, 'r', encoding='utf-8') as result:
    lines = result.readlines()
    for i, line in enumerate(lines):
        if 'Test' in line:
            if i < 60:
                train_acc_1.append(float(line.split('=')[2].split(',')[0]))
                train_f1_1.append(float(line.split('=')[3].split(',')[0]))
            elif i < 120:
                train_acc_2.append(float(line.split('=')[2].split(',')[0]))
                train_f1_2.append(float(line.split('=')[3].split(',')[0]))
            elif i < 180:
                train_acc_3.append(float(line.split('=')[2].split(',')[0]))
                train_f1_3.append(float(line.split('=')[3].split(',')[0]))
            elif i < 240:
                train_acc_4.append(float(line.split('=')[2].split(',')[0]))
                train_f1_4.append(float(line.split('=')[3].split(',')[0]))

plt.plot(train_acc_1, label='0.1')
plt.plot(train_acc_2, label='0.01')
plt.plot(train_acc_3, label='0.001')
plt.plot(train_acc_4, label='0.0001')
plt.xlabel('Epochs')
plt.title('Learning Rate ~ Test Accuracy')
plt.legend()
plt.show()

plt.plot(train_f1_1, label='0.1')
plt.plot(train_f1_2, label='0.01')
plt.plot(train_f1_3, label='0.001')
plt.plot(train_f1_4, label='0.0001')
plt.xlabel('Epochs')
plt.title('Learning Rate ~ Test F1_score')
plt.legend()
plt.show()
