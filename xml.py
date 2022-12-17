import os
import numpy
import pandas as pd
import torch
from PIL import Image,ImageDraw
import xml.etree.ElementTree as ET
import seaborn as sns
import matplotlib.pyplot as plt
import torchvision


images_dir='djangouploads\djangouploads\files\covers'

sample_image = Image.open("djangouploads\djangouploads\files\covers\img20190615_21002923_5.jpg")

sample_image

with open("djangouploads/djangouploads/sample_input_dataset/img20190615_21002923_5.xml") as annot_file:
    print(''.join(annot_file.readlines()))
    

tree = ET.parse('djangouploads/djangouploads/sample_input_dataset/img20190615_21002923_5.xml')
root = tree.getroot()
names = []
for data in root.findall('object'):
    name =  data.find('name').text
    names.append(name)
print(names)   

#####-------------------treee to xmin xmax and ymax ,ymin
tree = ET.parse('../input/bounding-boxes-xml/interview_task/sample_input_dataset/P00X000-2019092701422.xml')
root = tree.getroot()

sample_annotation = []

for neighbor in root.iter('bndbox'):
    xmin = int(neighbor.find('xmin').text)
    ymin = int(neighbor.find('ymin').text)
    xmax = int(neighbor.find('xmax').text)
    ymax = int(neighbor.find('ymax').text)
    
    sample_annotation.append([xmin,ymin,xmax,ymax])
print(sample_annotation)

#-------------------------------ImageDraw sample annotation------
sample_image_annotated = sample_image.copy()
img_bbox = ImageDraw.Draw(sample_image_annotated)

for bbox in sample_annotation:
    print(bbox)
    
    img_bbox.rectangle(bbox,outline="sign")
sample_image_annotated