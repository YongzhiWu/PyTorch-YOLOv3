# -*- coding: utf-8 -*-

from __future__ import print_function
import os
import numpy as np
import shutil

classes = {'bird': 0, 'car': 1, 'dog': 2, 'lizard': 3, 'turtle': 4}

def get_train_txt():
    os.makedirs("../data/tiny_vid/train/labels", exist_ok=True)
    os.makedirs("../data/tiny_vid/train/images", exist_ok=True)
    w = h = 128
    path = "../tiny_vid/"
    save_path = "../data/tiny_vid/train/"
    train_path = "data/tiny_vid/train/"
    img_list = []
    for classname in classes.keys():
        labels = np.loadtxt(path + classname + "_gt.txt").astype(np.int32).reshape(-1, 5)[:150,:]
        #print(labels[0])
        for i in range(labels.shape[0]):
            target = list(range(labels.shape[1]))
            #target = np.zeros((1, labels.shape[1]))
            target[0] = classes[classname]
            target[1] = np.float32((((labels[i][1] + labels[i][3])/2) / w))
            target[2] = np.float32(((labels[i][2] + labels[i][4])/2) / h)
            target[3] = np.float32((labels[i][3] - labels[i][1]) / w)
            target[4] = np.float32((labels[i][4] - labels[i][2]) / h)
            
            img_file = path + classname + "/" + "0"*(6 - len(str(labels[i][0]))) + str(labels[i][0]) + ".jpeg"
            
            img_path = save_path + "images/" + classname + "_" + str(labels[i][0]) + ".jpg"
            train_img_path = train_path + "images/" + classname + "_" + str(labels[i][0]) + ".jpg"
            target_path = save_path + "labels/" + classname + "_" + str(labels[i][0]) + ".txt"
            
            img_list.append(train_img_path)
            
            
            f = open(target_path, "w")
            for item in target:
                f.write(str(item))
                f.write(" ")
            f.close()
            
            #np.savetxt(target_path, target)
            #print(img_file)
            shutil.copy(img_file, img_path)
    f = open(save_path + "trainset.txt", "w")
    for item in img_list:
        f.write(item)
        f.write("\n")
    f.close()

def get_test_txt():
    os.makedirs("../data/tiny_vid/test/labels", exist_ok=True)
    os.makedirs("../data/tiny_vid/test/images", exist_ok=True)
    w = h = 128
    path = "../tiny_vid/"
    save_path = "../data/tiny_vid/test/"
    test_path = "data/tiny_vid/test/"
    img_list = []
    for classname in classes.keys():
        labels = np.loadtxt(path + classname + "_gt.txt").astype(np.int32).reshape(-1, 5)[150:180,:]
        #print(labels[0])
        for i in range(labels.shape[0]):
            target = list(range(labels.shape[1]))
            #target = np.zeros((1, labels.shape[1]))
            target[0] = classes[classname]
            target[1] = np.float32((((labels[i][1] + labels[i][3])/2) / w))
            target[2] = np.float32(((labels[i][2] + labels[i][4])/2) / h)
            target[3] = np.float32((labels[i][3] - labels[i][1]) / w)
            target[4] = np.float32((labels[i][4] - labels[i][2]) / h)
            
            img_file = path + classname + "/" + "0"*(6 - len(str(labels[i][0]))) + str(labels[i][0]) + ".jpeg"
            
            img_path = save_path + "images/" + classname + "_" + str(labels[i][0]) + ".jpg"
            test_img_path = test_path + "images/" + classname + "_" + str(labels[i][0]) + ".jpg"
            target_path = save_path + "labels/" + classname + "_" + str(labels[i][0]) + ".txt"
            
            img_list.append(test_img_path)
            
            
            f = open(target_path, "w")
            for item in target:
                f.write(str(item))
                f.write(" ")
            f.close()
            
            #np.savetxt(target_path, target)
            #print(img_file)
            shutil.copy(img_file, img_path)
    f = open(save_path + "testset.txt", "w")
    for item in img_list:
        f.write(item)
        f.write("\n")
    f.close()

if __name__ == "__main__":
    get_test_txt()