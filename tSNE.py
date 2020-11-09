# -*- coding: utf-8 -*-
import os
import numpy as np
import cv2
from time import time
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
# from sklearn import datasets #手写数据集要用到
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


#该函数是关键，需要根据自己的数据加以修改，将图片存到一个np.array里面，并且制作标签
#因为是两类数据，所以我分别用0,1来表示

# def get_data(dir_root_gt, target_seq_set, source_seq_set): #Input_path为你自己原始数据存储路径，我的路径就是上面的'./Images'
# # def get_data(dir_root_gt,source_seq_set):   
    
#     m = 0
#     data=np.zeros((18,1*1280*1024)) #初始化一个np.array数组用于存数据
#     print(data.shape)
#     label=np.ones((181,)) # Target domain

    
#     for i in target_seq_set:
#         frame_dir_temp = dir_root_gt + 'DomainAdaptation/' + str(i) + '/resized_frames/'
#         frame_dir_list = glob(frame_dir_temp + '/*.png')
#         total_frame = len(frame_dir_list)  # 149 
#         # print(total_frame)

#         # data=np.zeros((len(frame_dir_list),3*1280*1024)) #初始化一个np.array数组用于存数据
#         # print(data.shape)
#         # # label=np.zeros((len(Image_names),)) #初始化一个np.array数组用于存数据
#         # label=np.ones((len(frame_dir_list),)) # Target domain

#         for index in range(len(frame_dir_list)):
#             file_name = os.path.splitext(os.path.basename(frame_dir_list[index]))[0]
#             # image_path = os.path.join(dir_root_gt, str(i), "roi_features_resnet_all_ls/",file_name+"_node_features.npy")  # process .npy
#             image_path = os.path.join(dir_root_gt, 'DomainAdaptation/', str(i), "resized_frames",file_name+".png")   # process .png
#             # print(image_path)      
            
#             # img=np.load(image_path)   # load .npy file
#             img = cv2.imread(image_path)
#             img=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
#             # img=cv2.resize(img_gray,(1280,1024))
            
#             data[m]=img.ravel()
#             m += 1

#     print(m)


#     n = 0
#     data2=np.zeros((1124+392, 1*1280*1024)) #初始化一个np.array数组用于存数据
#     print(data2.shape)
#     label2=np.zeros((1124+392,)) # Target domain

    
#     # For all MICCAI iamges with caption, skip images without caption
#     for i in source_seq_set:
#         xml_dir_temp = dir_root_gt + 'seq_'+ str(i) + '/xml/'
#         # xml_dir_list = xml_dir_list + glob(xml_dir_temp + '/*.xml')   # 该路径就不是顺序的, 把新的seq路径放在之前的seq路径后面了,　因为是　xml_dir_list = xml_dir_list +　...
        
#         xml_dir_list = glob(xml_dir_temp + '/*.xml')   # 该路径就不是顺序的
#         # print(xml_dir_list)

#         # feature_dir_temp = dir_root_gt + str(i) + '/roi_features/'

#         random.shuffle(xml_dir_list)  # Should we shuffle here
        
#         total_xml = len(xml_dir_list)  # 149 
#         # print(total_xml)

#         for index in range(len(xml_dir_list)):
            
#             file_name = os.path.splitext(os.path.basename(xml_dir_list[index]))[0]
#             # print(file_name) # frame039
        
#             file_root = os.path.dirname(os.path.dirname(xml_dir_list[index]))
#             # print(file_root)  # /media/mmlab/data_2/mengya/instruments18_caption/seq_1
#             _xml = ET.parse(xml_dir_list[index]).getroot()
#             # print(_xml.find('caption_hard')) # None or True
            
#             if _xml.find('caption_hard') is None:
#                 # id_path = os.path.join("seq_"+str(i),"roi_features_resnet_all_ls/",file_name+"_node_features.npy")
#                 # No_caption.append(id_path)
#                 continue    
            
#             id_path = os.path.join("seq_"+str(i),"left_frames",file_name+".png")
#             image_path = os.path.join(dir_root_gt, id_path)
#             # print(image_path)
#             img = cv2.imread(image_path)
#             img=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
#             data2[n]=img.ravel()
#             n += 1

#     '''
#     # For all MICCAI iamges with and without caption
#     for i in source_seq_set:
#         frame_dir_temp = dir_root_gt + 'seq_'+str(i) + '/left_frames/'
#         frame_dir_list = glob(frame_dir_temp + '/*.png')
#         total_frame = len(frame_dir_list)  # 149 
#         # print(total_frame)

#         # data=np.zeros((len(frame_dir_list),3*1280*1024)) #初始化一个np.array数组用于存数据
#         # print(data.shape)
#         # # label=np.zeros((len(Image_names),)) #初始化一个np.array数组用于存数据
#         # label=np.ones((len(frame_dir_list),)) # Target domain

#         for index in range(len(frame_dir_list)):
#             file_name = os.path.splitext(os.path.basename(frame_dir_list[index]))[0]
#             # image_path = os.path.join(dir_root_gt, str(i), "roi_features_resnet_all_ls/",file_name+"_node_features.npy")  # process .npy
#             image_path = os.path.join(dir_root_gt, 'seq_'+str(i), "left_frames",file_name+".png")   # process .png
#             # print(image_path)      
            
#             # img=np.load(image_path)   # load .npy file
#             img = cv2.imread(image_path)
#             # img_gray=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
#             # img=cv2.resize(img_gray,(1280,1024))
            
#             data2[n]=img.ravel()
#             n += 1
#       '''      
    
#     print(n)
#     # np.vstack((a,b))
#     #  np.concatenate((a,b),axis=0)
#     data3 = np.vstack((data, data2))
#     label3 = np.hstack((label, label2))
#     print(data3.shape)
#     print(label3.shape)
#     return data3, label3

def get_data(dir_root_gt, target_seq_set, source_seq_set): #Input_path为你自己原始数据存储路径，我的路径就是上面的'./Images'
# def get_data(dir_root_gt,source_seq_set):   
    
    m = 0
    data=np.zeros((133,6*512)) #初始化一个np.array数组用于存数据
    print(data.shape)
    label=np.ones((133,)) # Target domain
    label = label.astype(int)

    
    for i in target_seq_set:
        frame_dir_temp = dir_root_gt + 'DomainAdaptation/' + str(i) + '/xml/'
        frame_dir_list = glob(frame_dir_temp + '/*.xml')
        total_frame = len(frame_dir_list)  # 149 
        # print(total_frame)

        # data=np.zeros((len(frame_dir_list),3*1280*1024)) #初始化一个np.array数组用于存数据
        # print(data.shape)
        # # label=np.zeros((len(Image_names),)) #初始化一个np.array数组用于存数据
        # label=np.ones((len(frame_dir_list),)) # Target domain

        for index in range(len(frame_dir_list)):
            file_name = os.path.splitext(os.path.basename(frame_dir_list[index]))[0]
            # image_path = os.path.join(dir_root_gt, str(i), "roi_features_resnet_all_ls/",file_name+"_node_features.npy")  # process .npy
            image_path = os.path.join(dir_root_gt, 'DomainAdaptation/', str(i), "roi_features_resnet_all_ls/",file_name+"_node_features.npy")   # process .png
            # print(image_path)      
            
            img=np.load(image_path)   # load .npy file
            # print(img)
            # img = cv2.imread(image_path)
            # img=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
            # img=cv2.resize(img_gray,(1280,1024))
            

            delta = 6 - img.shape[0]
            if delta > 0:
                img = np.concatenate([img, np.zeros((delta, img.shape[1]))], axis=0)  
            elif delta < 0:
                img = precomp_data[:6]

            data[m]=img.ravel()
            m += 1

    print(m)


    n = 0
    data2=np.zeros((1124, 6*512)) #初始化一个np.array数组用于存数据
    print(data2.shape)
    label2=np.zeros((1124,)) # Target domain
    label2 = label2.astype(int)
    
    # For all MICCAI iamges with caption, skip images without caption
    for i in source_seq_set:
        xml_dir_temp = dir_root_gt + 'seq_'+ str(i) + '/xml/'
        # xml_dir_list = xml_dir_list + glob(xml_dir_temp + '/*.xml')   # 该路径就不是顺序的, 把新的seq路径放在之前的seq路径后面了,　因为是　xml_dir_list = xml_dir_list +　...
        
        xml_dir_list = glob(xml_dir_temp + '/*.xml')   # 该路径就不是顺序的
        # print(xml_dir_list)

        # feature_dir_temp = dir_root_gt + str(i) + '/roi_features/'

        random.shuffle(xml_dir_list)  # Should we shuffle here
        
        total_xml = len(xml_dir_list)  # 149 
        # print(total_xml)

        for index in range(len(xml_dir_list)):
            
            file_name = os.path.splitext(os.path.basename(xml_dir_list[index]))[0]
            # print(file_name) # frame039
        
            file_root = os.path.dirname(os.path.dirname(xml_dir_list[index]))
            # print(file_root)  # /media/mmlab/data_2/mengya/instruments18_caption/seq_1
            _xml = ET.parse(xml_dir_list[index]).getroot()
            # print(_xml.find('caption_hard')) # None or True
            
            if _xml.find('caption_hard') is None:
                # id_path = os.path.join("seq_"+str(i),"roi_features_resnet_all_ls/",file_name+"_node_features.npy")
                # No_caption.append(id_path)
                continue    
            
            id_path = os.path.join("seq_"+str(i),"roi_features_resnet_all_ls/",file_name+"_node_features.npy")
            image_path = os.path.join(dir_root_gt, id_path)
            # print(image_path)
            # img = cv2.imread(image_path)
            # img=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
            img=np.load(image_path)



            delta = 6 - img.shape[0]
            if delta > 0:
                img = np.concatenate([img, np.zeros((delta, img.shape[1]))], axis=0)  
            elif delta < 0:
                img = precomp_data[:6]
            
            data2[n]=img.ravel()
            n += 1


    print(n)
    # np.vstack((a,b))
    #  np.concatenate((a,b),axis=0)
    data3 = np.vstack((data, data2))
    label3 = np.hstack((label, label2))
    print(data3.shape)
    print(label3.shape)
    return data3, label3

'''
下面的两个函数，
一个定义了二维数据，一个定义了3维数据的可视化
不作详解，也无需再修改感兴趣可以了解matplotlib的常见用法
'''
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

#主函数
def main():
    source_seq_set = [2, 3, 4, 6, 7, 9, 10, 11, 12, 14, 15] # train 1124
    target_seq_set = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13] # val 133

    # source_seq_set = [1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 14, 15, 16]
    # target_seq_set = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20] 
     
   
   
    dir_root_gt = '/media/mmlab/data_2/mengya/instruments18_caption/'
    
    data, label = get_data(dir_root_gt, target_seq_set, source_seq_set) #根据自己的路径合理更改
    # data, label = get_data(dir_root_gt, source_seq_set)
    
    
    print('Begining......') #时间会较长，所有处理完毕后给出finished提示
    tsne_2D = TSNE(n_components=2, init='pca', random_state=0) #调用TSNE
    result_2D = tsne_2D.fit_transform(data)
    
    tsne_3D = TSNE(n_components=3, init='pca', random_state=0)
    result_3D = tsne_3D.fit_transform(data)
    
    print('Finished......')
    # 调用上面的两个函数进行可视化
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