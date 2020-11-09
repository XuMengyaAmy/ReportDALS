# -*- coding: utf-8 -*-
import os
import numpy as np
import cv2
from time import time
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.manifold import TSNE
import pylab
from glob import glob


import sys
import json
import random

if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET



def get_data(dir_root_gt, target_seq_set, source_seq_set): 
   
    m = 0
    data=np.zeros((133,6*512)) 
    print(data.shape)
    label=np.ones((133,)) 
    label = label.astype(int)

    
    for i in target_seq_set:
        frame_dir_temp = dir_root_gt + 'DomainAdaptation/' + str(i) + '/xml/'
        frame_dir_list = glob(frame_dir_temp + '/*.xml')
        total_frame = len(frame_dir_list)  

        for index in range(len(frame_dir_list)):
            file_name = os.path.splitext(os.path.basename(frame_dir_list[index]))[0]      
            image_path = os.path.join(dir_root_gt, 'DomainAdaptation/', str(i), "roi_features_resnet_all_ls/",file_name+"_node_features.npy")      
            img=np.load(image_path)   
          
            

            delta = 6 - img.shape[0]
            if delta > 0:
                img = np.concatenate([img, np.zeros((delta, img.shape[1]))], axis=0)  
            elif delta < 0:
                img = precomp_data[:6]

            data[m]=img.ravel()
            m += 1

    print(m)


    n = 0
    data2=np.zeros((1124, 6*512)) 
    print(data2.shape)
    label2=np.zeros((1124,)) 
    label2 = label2.astype(int)
    
   
    for i in source_seq_set:
        xml_dir_temp = dir_root_gt + 'seq_'+ str(i) + '/xml/'
        xml_dir_list = glob(xml_dir_temp + '/*.xml')  
        random.shuffle(xml_dir_list)   
        total_xml = len(xml_dir_list) 
      
        for index in range(len(xml_dir_list)):     
            file_name = os.path.splitext(os.path.basename(xml_dir_list[index]))[0]    
            file_root = os.path.dirname(os.path.dirname(xml_dir_list[index]))   
            _xml = ET.parse(xml_dir_list[index]).getroot()

            if _xml.find('caption_hard') is None:
                continue    
            
            id_path = os.path.join("seq_"+str(i),"roi_features_resnet_all_ls/",file_name+"_node_features.npy")
            image_path = os.path.join(dir_root_gt, id_path)
            img=np.load(image_path)



            delta = 6 - img.shape[0]
            if delta > 0:
                img = np.concatenate([img, np.zeros((delta, img.shape[1]))], axis=0)  
            elif delta < 0:
                img = precomp_data[:6]
            
            data2[n]=img.ravel()
            n += 1


    print(n)

    data3 = np.vstack((data, data2))
    label3 = np.hstack((label, label2))
    print(data3.shape)
    print(label3.shape)
    return data3, label3


def plot_embedding_2D(data, label, title):
    x_min, x_max = np.min(data, 0), np.max(data, 0)
    data = (data - x_min) / (x_max - x_min)
    fig = plt.figure()
    for i in range(data.shape[0]):
        plt.text(data[i, 0], data[i, 1], str(label[i]),
                 color=plt.cm.Set1(label[i]),
                 fontdict={'weight': 'bold', 'size': 9})
    plt.xticks([])
    plt.yticks([])
    plt.title(title)
    plt.show()
    # return fig
def plot_embedding_3D(data,label,title): 
    x_min, x_max = np.min(data,axis=0), np.max(data,axis=0) 
    data = (data- x_min) / (x_max - x_min) 
    fig = plt.figure()
    ax = plt.figure().add_subplot(111,projection='3d') 
    for i in range(data.shape[0]): 
        ax.text(data[i, 0], data[i, 1], data[i,2],str(label[i]), color=plt.cm.Set1(label[i]),fontdict={'weight': 'bold', 'size': 9}) 
    plt.show()
    # return fig


def main():
    source_seq_set = [2, 3, 4, 6, 7, 9, 10, 11, 12, 14, 15] # train 1124
    target_seq_set = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13] # val 133

    dir_root_gt = '/media/mmlab/data_2/mengya/instruments18_caption/'
    
    data, label = get_data(dir_root_gt, target_seq_set, source_seq_set) 
    
    
    print('Begining......') 
    tsne_2D = TSNE(n_components=2, init='pca', random_state=0) 
    result_2D = tsne_2D.fit_transform(data)
    
    tsne_3D = TSNE(n_components=3, init='pca', random_state=0)
    result_3D = tsne_3D.fit_transform(data)
    
    print('Finished......')

    # fig1 = plot_embedding_2D(result_2D, label,'t-SNE')
    plot_embedding_2D(result_2D, label,'t-SNE')
    # plt.show(fig1)
    # pylab.show()

    # fig2 = plot_embedding_3D(result_3D, label,'t-SNE')
    plot_embedding_3D(result_3D, label,'t-SNE')
    # plt.show(fig2)
    # pylab.show()

if __name__ == '__main__':
    main()