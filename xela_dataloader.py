import os
import glob

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import h5py
import deepdish as dd
from PIL import Image
import csv
import numpy as np
from time import sleep
import cv2
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

# class MySampler(torch.utils.data.Sampler):
#     def __init__(self, end_idx, seq_length):
#         indices = []
#         for i in range(len(end_idx) - 1):
#             start = end_idx[i]
#             end = end_idx[i + 1] - seq_length
#             indices.append(torch.arange(start, end))
#         indices = torch.cat(indices)
#         self.indices = indices
#
#     def __iter__(self):
#         indices = self.indices[torch.randperm(len(self.indices))]
#         return iter(indices.tolist())
#
#     def __len__(self):
#         return len(self.indices)
def labels2cat(label_encoder, list):
    return label_encoder.transform(list)

def labels2onehot(OneHotEncoder, label_encoder, list):
    return OneHotEncoder.transform(label_encoder.transform(list).reshape(-1, 1)).toarray()

def onehot2labels(label_encoder, y_onehot):
    return label_encoder.inverse_transform(np.where(y_onehot == 1)[1]).tolist()

def cat2labels(label_encoder, y_cat):
    return label_encoder.inverse_transform(y_cat).tolist()
def load_data(path,clas,init,length,log):


    num=0
    # csvFile_train = open("csv_files/xeladataset_" +clas+'_'+str(length)+'_'+str(log)+".csv", 'w+')
    # csvFile_test=open("xela_deligrasp_test" + str(frame) +'_'+str(log)+ '_'+str(train_val_radio) +".csv", 'w')
    # writer_train= csv.writer(csvFile_train)

    # nestl1=csv.reader(open(clas+'_new.csv','r'))
    caselist=os.listdir(path+"/"+clas+'/')
    # writer_train = csv.writer(csvFile_train)
    # writer_test=csv.writer(csvFile_test)

    # dataset_all=[]
    # for row in nestl1:
    #     dataset_all.append(row)
    # dataset_train,dataset_test=train_test_split(dataset_all,test_size=1-train_val_radio)
    cases=[]
    for case in caselist:
        # print(case)
        # rowTemßp=[]
        pathTemp_visual=path+"/"+clas+'/'+case+'/'+'visual/'
        pathTemp_tactile=path+"/"+clas+'/'+case+'/'+'tactile/'
        time_list_visual=np.load(pathTemp_visual+'visual_time_list.npy')
        time_lst_tactile=np.load(pathTemp_tactile+'tactile_time_list.npy')
        num_visual=len(os.listdir(pathTemp_visual))-1
        num_tactile=len(os.listdir(pathTemp_tactile))-1
        width,force,label=case.split("_")
        # print(num_visual,num_tactile)
        for i in range(init,num_visual-length-1,log):
            rowTemp=[]
            # print(i)
            rowTemp.append(width)
            rowTemp.append(force)
            rowTemp.append(label)
            for k in range(length):
                rowTemp.append(pathTemp_visual+str(i+k)+'.jpg')
                # print(path+case,i+k)
            tactile_time_length=0
            for j in range(num_tactile):
                if time_lst_tactile[j] > time_list_visual[i] and time_lst_tactile[j] < time_list_visual[i+length]:
                    rowTemp.append(pathTemp_tactile+str(j)+'.jpg')
                    tactile_time_length+=1
            rowTemp.append(tactile_time_length)
            cases.append(rowTemp)
    return cases
            # writer_train.writerow(rowTemp)
    # csvFile_train.close()
def train_test_dataset(path,visual_seq_length,tactile_seq_length,log,flag):
    appbox=load_data(path, 'appbox', 10, visual_seq_length, log)
    baisui=load_data(path, 'baisui', 10, visual_seq_length, log)
    bingho=load_data(path, 'bingho', 10, visual_seq_length, log)
    cesbon=load_data(path, 'cesbon', 10, visual_seq_length, log)
    cokele=load_data(path, 'cokele', 5, visual_seq_length, log)
    haitun=load_data(path, 'jianjo', 5, visual_seq_length, log)
    jianjo=load_data(path, 'meinad', 5, visual_seq_length, log)
    nongf1=load_data(path, 'nongf1', 5, visual_seq_length, log)
    # load_data('/workspace/csw/graspingdata','nongfu',5,5,1)
    pacup1=load_data(path, 'pacup1', 5, visual_seq_length, log)
    pacup2=load_data(path, 'pacup2', 5, visual_seq_length, log)
    songsu=load_data(path, 'songsu', 7, visual_seq_length, log)
    zhijin=load_data(path, 'zhijin', 4, visual_seq_length, log)
    train_dataset=appbox+baisui+bingho+cokele+haitun+jianjo+pacup1+pacup2+zhijin
    test_dataset=cesbon+nongf1+songsu
    if flag == 'train':
        dataset=train_dataset
    elif flag == 'test':
        dataset=test_dataset
    return dataset

class MyDataset(Dataset):
    def __init__(self, image_paths, visual_seq_length, tactile_seq_length,transform_v,transform_t,log,flag):
        self.image_paths = image_paths
        self.visual_seq_length = visual_seq_length
        self.tactile_seq_length = tactile_seq_length
        self.transform_v = transform_v
        self.transform_t = transform_t
        # self.csvReader=csv.reader(open(image_paths))
        self.label=[]
        self.visual_sequence=[]
        self.tactile_sequence=[]
        self.classes=['0','1','2']
        self.log=log
        self.flag=flag
        self.dataset=train_test_dataset(self.image_paths,self.visual_seq_length,self.tactile_seq_length,self.log,self.flag)
        # self.tactile_sequence_length=[]
        le = LabelEncoder()
        le.fit(self.classes)

# convert category -> 1-hot
        action_category = le.transform(self.classes).reshape(-1, 1)
        enc = OneHotEncoder()
        enc.fit(action_category)
        for item in self.dataset:
            self.label.append(str(item[2]))
            # self.tactile_sequence_length.append(int(item[-1]))
            visual = []
            tactile=[]
            for i in range(self.visual_seq_length):
                visual.append(item[3+i])
            for j in range(self.tactile_seq_length):
                tactile.append(item[j+3+self.visual_seq_length])
            self.visual_sequence.append(visual)
            self.tactile_sequence.append(tactile)
        self.label=labels2cat(le, self.label)
        #print(len(self.image_sequence))
    def __getitem__(self, index):

        visuals = []
        tactiles=[]
        for i in range(self.visual_seq_length):
            visualTemp=Image.open(self.visual_sequence[index][i])
            if self.transform_v:
                visualTemp = self.transform_v(visualTemp)
            visuals.append(visualTemp.unsqueeze(1))
        for j in range(self.tactile_seq_length):
            tactileTemp=Image.open(self.tactile_sequence[index][j])
            if self.transform_t:
                tactileTemp = self.transform_t(tactileTemp)
                # print(tactileTemp.shape)
            tactiles.append(tactileTemp.unsqueeze(1))


        x_v = torch.cat(visuals,dim=1)
        x_t=torch.cat(tactiles,dim=1)
        # print(x_v.shape,x_t.shape)

        y = torch.tensor(self.label[index], dtype=torch.long)
        # print(x_v.shape,x_t.shape,y)
        return x_v,x_t, y

    def __len__(self):
        return len(self.visual_sequence)



