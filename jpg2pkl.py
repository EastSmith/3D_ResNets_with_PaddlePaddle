###########jpg2pkl.py#########
import os
import numpy as np
import cv2
import sys
import glob
import pickle
from multiprocessing import Pool


label_dic = np.load('label_dir.npy', allow_pickle=True).item()
print(label_dic)

source_dir = 'data1/UCF-101_JPG'          ###
target_train_dir = 'data1/UCF-101_JPG/train'      ###
target_test_dir = 'data1/UCF-101_JPG/test'         
target_val_dir = 'data1/UCF-101_JPG/val'
if not os.path.exists(target_train_dir):
    os.mkdir(target_train_dir)
if not os.path.exists(target_test_dir):
    os.mkdir(target_test_dir)
if not os.path.exists(target_val_dir):
    os.mkdir(target_val_dir)

for key in label_dic:
    each_mulu = key + '_jpg'
    # each_mulu = key 
    print(each_mulu, key)

    label_dir = os.path.join(source_dir, each_mulu)
    label_mulu = os.listdir(label_dir)
    tag = 1
    # print(label_dir)
    # print(len(label_mulu))
    for each_label_mulu in label_mulu:
        image_file = os.listdir(os.path.join(label_dir, each_label_mulu))
        image_file.sort()
        image_name = image_file[0][:-6]
        image_num = len(image_file)
        frame = []
        vid = image_name
        # print(vid)
        sp = int(vid[-6:-4])   ####根据ucf101的split01，通过文件夹名字划分数据集
        # print(sp)
        for i in range(image_num):
            image_path = os.path.join(os.path.join(label_dir, each_label_mulu), image_name + '_' + str(i+1) + '.jpg')
            frame.append(image_path)

        output_pkl = vid + '.pkl'
        if sp >= 8:
            output_pkl = os.path.join(target_train_dir, output_pkl)
        
        else:
            output_pkl = os.path.join(target_val_dir, output_pkl)
        tag += 1
        f = open(output_pkl, 'wb')
        pickle.dump((vid, label_dic[key], frame), f, -1)
        f.close()
