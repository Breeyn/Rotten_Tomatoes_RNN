import torch
import random
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pack_sequence, pad_packed_sequence
import gzip
import csv
import time
import pandas as pd
import os

# Parameters
HIDDEN_SIZE = 70
BATCH_SIZE = 100
N_LAYER = 3
N_EPOCHS = 30
N_WORDS = 40000
N_LABELS = 5
USE_GPU = True
N_TRAIN = 156060
PART = 140454#训练集占比0.9,测试集占比0.1
dic={}
N_TEST = 66292
cnt=1

class SentDataset(Dataset):
    def __init__(self,part,file, is_train_set=True):
        f=open(file,'r')
        self.sents=[]
        self.labels = []
        reader = csv.reader(f,delimiter='\t')#读取数据
        rows = list(reader)
        del rows[0]

        if is_train_set:#读入训练集
            for i in range(part):
                self.sents.append(rows[i][2])
                self.labels.append(int(rows[i][3]))
            self.len=len(self.sents)
            self.label_num = len(self.labels)

        else:#读入测试集
            for i in range(part,N_TRAIN):
                self.sents.append(rows[i][2])
                self.labels.append(int(rows[i][3]))
            self.len=len(self.sents)
            self.label_num = len(self.labels)

    def __getitem__(self, index):
        return self.sents[index], self.labels[index]

    def __len__(self):
        return self.len


class SentTestset(Dataset):
    def __init__(self,file):
        f=open(file,'r')
        self.sents=[]
        self.len=0
        reader = csv.reader(f,delimiter='\t')#读取数据
        rows = list(reader)
        del rows[0]

        for i in range(N_TEST):
            self.sents.append(rows[i][2])

        self.len=len(self.sents)

    def __getitem__(self, index):
        return self.sents[index]

    def __len__(self):
        return len(self.sents)

#RNN分类器
class RNNClassifier(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_layers=1, bidirectional=True):
        super(RNNClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.n_directions = 2 if bidirectional else 1

        self.embedding = torch.nn.Embedding(input_size, hidden_size)
        self.gru = torch.nn.GRU(hidden_size, hidden_size, n_layers,
                                bidirectional=bidirectional)
        self.fc = torch.nn.Linear(hidden_size * self.n_directions, output_size)

    def _init_hidden(self, batch_size):
        hidden = torch.zeros(self.n_layers * self.n_directions,
                            batch_size, self.hidden_size)
        return create_tensor(hidden)


    def forward(self, input, seq_lengths):
 
        input = input.t()
        batch_size = input.size(1)

        hidden = self._init_hidden(batch_size)
        embedding = self.embedding(input)

        output, hidden = self.gru(embedding, hidden)
        if self.n_directions == 2:
            hidden_cat = torch.cat([hidden[-1], hidden[-2]], dim=1)
        else:
            hidden_cat = hidden[-1]

        fc_output = self.fc(hidden_cat)

        return fc_output


def tensors_train(sents, labels):
    labels=torch.Tensor(labels)
    sequences_and_lengths = [sent2list(sent) for sent in sents]
    sent_sequences = [sl[0] for sl in sequences_and_lengths]
    seq_lengths = torch.LongTensor([sl[1] for sl in sequences_and_lengths])
    labels = labels.long()

    seq_tensor = torch.zeros(len(sent_sequences), seq_lengths.max()).long()
    for idx, (seq, seq_len) in enumerate(zip(sent_sequences, seq_lengths), 0):
        seq_tensor[idx, :seq_len] = torch.LongTensor(seq)
    return create_tensor(seq_tensor), \
           create_tensor(seq_lengths),\
           create_tensor(labels)

def tensors_test(sents):
    sequences_and_lengths = [sent2list(sent) for sent in sents]
    sent_sequences = [sl[0] for sl in sequences_and_lengths]
    seq_lengths = torch.LongTensor([sl[1] for sl in sequences_and_lengths])

    seq_tensor = torch.zeros(len(sent_sequences), seq_lengths.max()).long()
    for idx, (seq, seq_len) in enumerate(zip(sent_sequences, seq_lengths), 0):
        seq_tensor[idx, :seq_len] = torch.LongTensor(seq)
    return create_tensor(seq_tensor), \
           create_tensor(seq_lengths)


def sent2list(sent):#将句子转换成由单词序列组成的列表
    arr=[]
    for w in sent.split(' '):
        w=w.lower()
        if w in dic.keys():
            arr.append(dic[w])
    #arr = [dic[w] for w in sent.split(' ')]
    return arr, len(arr)

#创建tensor
def create_tensor(tensor):
    if USE_GPU:
        device = torch.device("cuda:0")
        tensor = tensor.to(device)
    return tensor

#训练模型
def trainModel():
    total_loss = 0
    for i, (sents, labels) in enumerate(trainloader, 1):
        inputs, seq_lengths, target = tensors_train(sents, labels)
        output = classifier(inputs, seq_lengths)
        loss = criterion(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        if i % 10 == 0:
            print(f'[{i * len(inputs)}/{len(trainset)}] ', end='')
            print(f'loss={total_loss / (i * len(inputs))}')
            
    return total_loss

#测试准确率
def testModel():
    correct = 0
    total = len(testset)
    print("evaluating trained model ...")
    with torch.no_grad():
        for i, (sents, labels) in enumerate(testloader, 1):
            inputs, seq_lens, target = tensors_train(sents, labels)
            output = classifier(inputs, seq_lens)
            pred = output.max(dim=1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

        percent = '%.2f' % (100 * correct / total)
        print(f'Test set: Accuracy {correct}/{total} {percent}%')

    return correct / total

def test():
    #total = len(verifyset)
    f1 = open('python_work\\data\\submission.csv', 'w')
    cnt=156061
    with torch.no_grad():
        for i, (sents) in enumerate(verifyloader, 1):
            inputs,seq_lens=tensors_test(sents)
            output = classifier(inputs, seq_lens)
            pred = output.max(dim=1, keepdim=True)[1]
            lis1=pred.tolist()
            for e in lis1:
                # e.insert(0,str(cnt))
                f1.writelines(str(cnt)+','+str(e[0])+os.linesep)
                # f1.write(',')
                # f1.write(str(e[0])+os.linesep)
                cnt=cnt+1
            # print(output,type(output),len(output))
    f1.close()

#随机打乱划分训练集和测试集
def randomly(filein,fileout):
    file_out = open(fileout,'a', encoding='gb18030', errors='ignore')
    lines = []
    with open(filein, 'r', encoding='gb18030', errors='ignore') as f: 
        for line in f:  
            lines.append(line)
        
    del lines[0]
    random.shuffle(lines)
    for line in lines:
        file_out.writelines(line)
    file_out.close()

#将出现过的所有单词存入字典
def prepare(filename,num):
    global cnt
    f=open(filename,'r')

    for i in range(num+1):
        line=f.readline()
        if i == 0 :
            continue
        
        for word in line.split('\t')[2].split(' '):
            word=word.lower()
            if word not in dic.keys():
                cnt=cnt+1
                dic[word]=cnt


if __name__ == '__main__':

    filein = 'data\\train.tsv'
    fileout = 'data\\train1.tsv'
    fileverify='data\\test.tsv'

    prepare(filein,N_TRAIN)
    prepare(fileverify,N_TEST)

    print("Training for %d epochs..." % N_EPOCHS)
    classifier = RNNClassifier(N_WORDS, HIDDEN_SIZE, N_LABELS, N_LAYER)
    if USE_GPU:
        device = torch.device("cuda:0")
        classifier.to(device)

    criterion = torch.nn.CrossEntropyLoss()
    # optimizer = torch.optim.Adam(classifier.parameters(), lr=0.0035)#此处调整学习率
    optimizer = torch.optim.Adam(classifier.parameters(), lr=0.004)#此处调整学习率
#110,0.005   67.01
#90,0.003    67.38
#90,0.004    67.46
#90,0.001    67.04
#80,0.003    67.12
#70,0.0035   67.49
#70,0.003    67.39
#70,0.004    67.34
#70,0.002    66.81
#90,0.004  Adamax  66.56
#90,0.001    66.59
    acc_list = []#记录准确率
    # randomly(filein,fileout)#打乱数据
    verifyset = SentTestset(fileverify)#读入验证集
    verifyloader = DataLoader(verifyset, batch_size=BATCH_SIZE, shuffle=False)

    trainset = SentDataset(PART,fileout,is_train_set=True)#读入训练集
    trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)
    testset = SentDataset(PART,fileout,is_train_set=False)#读入测试集
    testloader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False)

    


    for epoch in range(1, N_EPOCHS + 1):
        print('epoch: %d'%epoch)
        trainModel()#训练模型
        acc = testModel()#测试
        acc_list.append(acc)#记录准确率
    test()
    

