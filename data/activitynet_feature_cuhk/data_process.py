# -*- coding: utf-8 -*-

import random
import numpy as np
import scipy
import pandas as pd
import pandas
import numpy
import json


def resizeFeature(inputData,newSize):
    # inputX: (temporal_length,feature_dimension) #
    originalSize=len(inputData)
    #print originalSize
    if originalSize==1:
        inputData=np.reshape(inputData,[-1])
        return np.stack([inputData]*newSize)
    x=numpy.array(range(originalSize))
    f=scipy.interpolate.interp1d(x,inputData,axis=0)
    x_new=[i*float(originalSize-1)/(newSize-1) for i in range(newSize)]
    y_new=f(x_new)
    return y_new

def readData(video_name,data_type=["spatial","temporal"]):
    spatial_dir="./spatial/csv_action/"
    temporal_dir="./temporal/csv_action/"
    data=[]
    for dtype in data_type:
        if dtype=="spatial":
            df=pandas.read_csv(spatial_dir+video_name+".csv")
        elif dtype=="temporal":
            df=pandas.read_csv(temporal_dir+video_name+".csv")
        data.append(df.values[:,:])
    lens=[len(d) for d in data]
    #print lens
    min_len=min(lens)
    new_data=[d[:min_len] for d in data]
    new_data=numpy.concatenate(new_data,axis=1)
    return new_data

def load_json(file):
    with open(file) as json_file:
        data = json.load(json_file)
        return data

def getDatasetDict():
    df=pd.read_csv("./info/video_info.csv")
    json_data= load_json("./info/activity_net.v1-3.min.json")
    database=json_data['database']
    out_dict={}
    for i in range(len(df)):
        video_name=df.video.values[i]
        video_info=database[video_name[2:]]
        video_new_info={}
        video_new_info['duration_frame']=df.numFrame.values[i]
        video_new_info['duration_second']=df.seconds.values[i]
        video_new_info['annotations']=video_info['annotations']
        out_dict[video_name]=video_new_info
    return out_dict

def poolData(data,videoAnno,num_prop=100,num_bin=1,num_sample_bin=3,pool_type="mean"):
    feature_frame=len(data)*16
    video_frame=videoAnno['duration_frame']
    video_second=videoAnno['duration_second']
    corrected_second=float(feature_frame)/video_frame*video_second
    fps=float(video_frame)/video_second
    st=16/fps

    if len(data)==1:
        video_feature=np.stack([data]*num_prop)
        video_feature=np.reshape(video_feature,[num_prop,400])
        return video_feature

    x=[st/2+ii*st for ii in range(len(data))]
    f=scipy.interpolate.interp1d(x,data,axis=0)
        
    video_feature=[]
    zero_sample=np.zeros(num_bin*400)
    tmp_anchor_xmin=[1.0/num_prop*i for i in range(num_prop)]
    tmp_anchor_xmax=[1.0/num_prop*i for i in range(1,num_prop+1)]        
    
    num_sample=num_bin*num_sample_bin
    for idx in range(num_prop):
        xmin=max(x[0]+0.0001,tmp_anchor_xmin[idx]*corrected_second)
        xmax=min(x[-1]-0.0001,tmp_anchor_xmax[idx]*corrected_second)
        if xmax<x[0]:
            #print "fuck"
            video_feature.append(zero_sample)
            continue
        if xmin>x[-1]:
            video_feature.append(zero_sample)
            continue
            
        plen=(xmax-xmin)/(num_sample-1)
        x_new=[xmin+plen*ii for ii in range(num_sample)]
        y_new=f(x_new)
        y_new_pool=[]
        for b in range(num_bin):
            tmp_y_new=y_new[num_sample_bin*b:num_sample_bin*(b+1)]
            if pool_type=="mean":
                tmp_y_new=np.mean(y_new,axis=0)
            elif pool_type=="max":
                tmp_y_new=np.max(y_new,axis=0)
            y_new_pool.append(tmp_y_new)
        y_new_pool=np.stack(y_new_pool)
        y_new_pool=np.reshape(y_new_pool,[-1])
        video_feature.append(y_new_pool)
    video_feature=np.stack(video_feature)
    return video_feature

videoDict=getDatasetDict()
videoNameList=videoDict.keys()
random.shuffle(videoNameList)
col_names=[]
for i in range(400):
    col_names.append("f"+str(i))

for videoName in videoNameList:
    videoAnno=videoDict[videoName]
    data=readData(videoName)
    numFrame=videoAnno['duration_frame']
    featureFrame=len(data)*16
    
    videoAnno["feature_frame"]=featureFrame
    videoDict[videoName]=videoAnno
    print numFrame,featureFrame
    
    videoFeature_mean=poolData(data,videoAnno,num_prop=100,num_bin=1,num_sample_bin=3,pool_type="mean")
    outDf=pd.DataFrame(videoFeature_mean,columns=col_names)
    outDf.to_csv("./csv_mean_100/"+videoName+".csv",index=False)


outfile=open("./anet_anno_anet.json","w")
json.dump(videoDict,outfile)
outfile.close()
