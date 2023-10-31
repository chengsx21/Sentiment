'''
    @author: chengsx21
    @file: pipeline.py
    @time: 2023/05/01
'''
import os

def pipeline():
    '''
        This function is used to run the pipeline
    '''
    print("--------------------")
    print("Result of TextCNN")
    print("--------------------")
    os.system("python main.py -p CNN")
    print("--------------------")
    print("Result of LstmRNN")
    print("--------------------")
    os.system("python main.py -p RNN1")
    print("--------------------")
    print("Result of GruRNN")
    print("--------------------")
    os.system("python main.py -p RNN2")
    print("--------------------")
    print("Result of MLP")
    print("--------------------")
    os.system("python main.py -p MLP")

if __name__ == "__main__":
    pipeline()
