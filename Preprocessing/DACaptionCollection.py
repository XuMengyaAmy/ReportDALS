import os
import sys
import random
import numpy as np
from glob import glob
from torch.utils.data import Dataset, DataLoader
import json

import h5py
import re

if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET



# seq_set = [14, 15, 16, 17, 18, 19, 20] # train 48
seq_set = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13] # val 133



xml_dir_list = []
dir_root_gt = '/media/mmlab/data_2/mengya/instruments18_caption/DomainAdaptation'
annotation = []  

No_caption = []
With_caption = []

features = []



for i in seq_set:
    xml_dir_temp = dir_root_gt + '/' + str(i) + '/xml/' 

    xml_dir_list = glob(xml_dir_temp + '/*.xml')   
 
    random.shuffle(xml_dir_list)  
    
    total_xml = len(xml_dir_list)  
    print(total_xml)

    for index in range(len(xml_dir_list)):
        
        file_name = os.path.splitext(os.path.basename(xml_dir_list[index]))[0]
       
     
        file_root = os.path.dirname(os.path.dirname(xml_dir_list[index]))
       
        _xml = ET.parse(xml_dir_list[index]).getroot()

        temp_anno = {}

        tem_fea = {}
        
        
        
        if _xml.find('caption_hard') is None:
            id_path = os.path.join(str(i),"roi_features_resnet_all_ls/",file_name+"_node_features.npy")
            No_caption.append(id_path)
            continue    
        temp_anno['id_path'] = os.path.join(str(i),"roi_features_resnet_all_ls/",file_name+"_node_features.npy")
        temp_anno['caption'] = _xml.find('caption_hard').text
        annotation.append(temp_anno)


        id_path = os.path.join(str(i),"roi_features_resnet_all_ls",file_name+"_node_features.npy")
        With_caption.append(id_path)

        
        node_features = np.load(os.path.join(file_root, 'roi_features_resnet_all_ls', '{}_node_features.npy').format(file_name))
        tem_fea['id_path'] = os.path.join(str(i),"roi_features_resnet_all_ls/",file_name+"_node_features.npy") 
       
        tem_fea['feature'] = node_features
        features.append(tem_fea)


if not os.path.exists('annotations/annotations_DA'):
    os.makedirs('annotations/annotations_DA')

with open('annotations/annotations_DA/captions_val.json', 'w') as f:
    json.dump(annotation, f)

with open('annotations/annotations_DA/NoCaption_id_path_val.json', 'w') as f:
    json.dump(No_caption, f)

with open('annotations/annotations_DA/WithCaption_id_path_val.json', 'w') as f:
    json.dump(With_caption, f)



print(len(annotation))
print(len(features))





   


   

