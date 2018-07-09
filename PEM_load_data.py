# -*- coding: utf-8 -*-
import random
import pandas
import numpy
import json

def load_json(file):
    with open(file) as json_file:
        data = json.load(json_file)
        return data

def getDatasetDict():
    df=pandas.read_csv("./data/activitynet_annotations/video_info_new.csv")
    json_data= load_json("./data/activitynet_annotations/anet_anno_action.json")
    database=json_data
    train_dict={}
    val_dict={}
    test_dict={}
    for i in range(len(df)):
        video_name=df.video.values[i]
        video_info=database[video_name]
        video_new_info={}
        video_new_info['duration_frame']=video_info['duration_frame']
        video_new_info['duration_second']=video_info['duration_second']
        video_new_info["feature_frame"]=video_info['feature_frame']
        video_subset=df.subset.values[i]
        video_new_info['annotations']=video_info['annotations']
        if video_subset=="training":
            train_dict[video_name]=video_new_info
        elif video_subset=="validation":
            val_dict[video_name]=video_new_info
        elif video_subset=="testing":
            test_dict[video_name]=video_new_info
    return train_dict,val_dict,test_dict

def getBatchList(video_dict,batch_size,shuffle=True):
    ## notice that there are some video appear twice in last two batch ##
    video_list=video_dict.keys()
    batch_start_list=[i*batch_size for i in range(len(video_list)/batch_size)]
    batch_start_list.append(len(video_list)-batch_size)
    if shuffle==True:
        random.shuffle(video_list)
    batch_video_list=[]
    for bstart in batch_start_list:
        batch_video_list.append(video_list[bstart:(bstart+batch_size)])
    return batch_video_list
    
def prop_dict_data(prop_dict):
    prop_name_list=prop_dict.keys()
    batch_feature=[]
    batch_iou_list=[]
    batch_ioa_list=[]
    for prop_name in prop_name_list:
        batch_feature.append(prop_dict[prop_name]["bsp_feature"])
        batch_iou_list.extend(list(prop_dict[prop_name]["match_iou"]))
        batch_ioa_list.extend(list(prop_dict[prop_name]["match_ioa"]))

    batch_feature=numpy.concatenate(batch_feature)
    return batch_feature,batch_iou_list,batch_ioa_list

def getProposalData(video_dict,video_list):
    prop_dict={}
    for video_name in video_list:
        pdf=pandas.read_csv("./output/PGM_proposals/"+video_name+".csv")
        pdf=pdf[:500]
        tmp_feature = numpy.load("./output/PGM_feature/" + video_name+".npy")
        tmp_feature = tmp_feature[:500]
        tmp_dict={"match_iou":pdf.match_iou.values[:],"match_ioa":pdf.match_ioa.values[:],
                  "xmin":pdf.xmin.values[:],"xmax":pdf.xmax.values[:],
                  "bsp_feature":tmp_feature}
        prop_dict[video_name]=tmp_dict
    return prop_dict
            
def getProposalDataTest(video_dict,video_name):
    pdf=pandas.read_csv("./output/PGM_proposals/"+video_name+".csv")
    pdf=pdf[:1000]
    tmp_feature = numpy.load("./output/PGM_feature/" + video_name+".npy")
    tmp_feature = tmp_feature[:1000]
    prop_dict={"match_iou":pdf.xmin.values[:],"match_ioa":pdf.xmin.values[:],
                  "xmin":pdf.xmin.values[:],"xmax":pdf.xmax.values[:],"xmin_score":pdf.xmin_score.values[:],"xmax_score":pdf.xmax_score.values[:],
                  "bsp_feature":tmp_feature}
    return prop_dict,video_name

def getTestData(dataSet):
    train_dict,val_dict,test_dict=getDatasetDict()

    if dataSet=="test":
        video_dict=test_dict
    else:
        video_dict=val_dict
    video_list=video_dict.keys()#[:500]
    
    FullData={}
    i=0
    for video_name in video_list:
        if i%100 == 0:
            print "%d / %d videos in %s set is loaded" %(i,len(video_list),dataSet) 
        i+=1
        prop_dict,video_name = getProposalDataTest(video_dict,video_name)
        FullData[video_name]=prop_dict
    return FullData

def getTrainData(batch_size,dataSet):
    train_dict,val_dict,test_dict=getDatasetDict()
    if dataSet=="validation":
        video_dict=val_dict
    else:
        video_dict=train_dict
    batch_video_list=getBatchList(video_dict,batch_size)
    
    FullData=[]
    i=0
    for video_list in batch_video_list:
        if i%10 == 0:
            print "%d / %d batch_data in %s set is loaded" %(i,len(batch_video_list),dataSet) 
        i+=1
        FullData.append(getProposalData(video_dict,video_list))
    return FullData