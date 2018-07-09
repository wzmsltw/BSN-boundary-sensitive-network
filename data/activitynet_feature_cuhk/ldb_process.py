# -*- coding: utf-8 -*-
"""
Created on Mon May 15 22:31:31 2017

@author: wzmsltw
"""
import caffe
import leveldb
import numpy as np
from caffe.proto import caffe_pb2
import pandas as pd

col_names=[]
for i in range(200):
    col_names.append("f"+str(i))

df=pd.read_table("./input_spatial_list.txt",names=['image','frame','label'],sep=" ")

db = leveldb.LevelDB('./LDB')
datum = caffe_pb2.Datum()

i=0
video_name="init"
videoData=np.reshape([],[-1,200])

for key, value in db.RangeIter():
    
    tmp_video_name=df.image.values[i].split('/')[-1]
    if tmp_video_name !=video_name:
        outDf=pd.DataFrame(videoData,columns=col_names)
        outDf.to_csv("./csv_raw/"+video_name+".csv",index=False)
        videoData=np.reshape([],[-1,200])
        video_name=tmp_video_name
    i+=1
    
    datum.ParseFromString(value)
    label = datum.label
    data = caffe.io.datum_to_array(datum)
    data=np.reshape(data,[1,200])
    videoData=np.concatenate((videoData,data))

del db
