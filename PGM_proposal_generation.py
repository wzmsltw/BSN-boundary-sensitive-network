# -*- coding: utf-8 -*-
import json
import numpy
import pandas
import argparse


def load_json(file):
    with open(file) as json_file:
        data = json.load(json_file)
        return data
        
def iou_with_anchors(anchors_min,anchors_max,box_min,box_max):
    """Compute jaccard score between a box and the anchors.
    """
    len_anchors=anchors_max-anchors_min
    int_xmin = numpy.maximum(anchors_min, box_min)
    int_xmax = numpy.minimum(anchors_max, box_max)
    
    inter_len = numpy.maximum(int_xmax - int_xmin, 0.)
  
    union_len = len_anchors - inter_len +box_max-box_min
    #print inter_len,union_len
    jaccard = numpy.divide(inter_len, union_len)
    return jaccard

def ioa_with_anchors(anchors_min,anchors_max,box_min,box_max):
    """Compute intersection between score a box and the anchors.
    """
    len_anchors=anchors_max-anchors_min
    int_xmin = numpy.maximum(anchors_min, box_min)
    int_xmax = numpy.minimum(anchors_max, box_max)
    inter_len = numpy.maximum(int_xmax - int_xmin, 0.)
    scores = numpy.divide(inter_len, len_anchors)
    return scores

def generateProposals(video_name,video_dict):
    tscale = 100    
    tgap = 1./tscale
    peak_thres=0.5

    tdf=pandas.read_csv("./output/TEM_results/"+video_name+".csv")
    start_scores=tdf.start.values[:]
    end_scores=tdf.end.values[:]
    
    max_start = max(start_scores)
    max_end = max(end_scores)
    
    start_bins=numpy.zeros(len(start_scores))
    start_bins[[0,-1]]=1
    for idx in range(1,tscale-1):
        if start_scores[idx]>start_scores[idx+1] and start_scores[idx]>start_scores[idx-1]:
            start_bins[idx]=1
        elif start_scores[idx]>(peak_thres*max_start):
            start_bins[idx]=1
                
    end_bins=numpy.zeros(len(end_scores))
    end_bins[[0,-1]]=1
    for idx in range(1,tscale-1):
        if end_scores[idx]>end_scores[idx+1] and end_scores[idx]>end_scores[idx-1]:
            end_bins[idx]=1
        elif end_scores[idx]>(peak_thres*max_end):
            end_bins[idx]=1
    
    xmin_list=[]
    xmin_score_list=[]
    xmax_list=[]
    xmax_score_list=[]
    for j in range(tscale):
        if start_bins[j]==1:
            xmin_list.append(tgap/2+tgap*j)
            xmin_score_list.append(start_scores[j])
        if end_bins[j]==1:
            xmax_list.append(tgap/2+tgap*j)
            xmax_score_list.append(end_scores[j])
            
    new_props=[]
    for ii in range(len(xmax_list)):
        tmp_xmax=xmax_list[ii]
        tmp_xmax_score=xmax_score_list[ii]
        
        for ij in range(len(xmin_list)):
            tmp_xmin=xmin_list[ij]
            tmp_xmin_score=xmin_score_list[ij]
            if tmp_xmin>=tmp_xmax:
                break
            new_props.append([tmp_xmin,tmp_xmax,tmp_xmin_score,tmp_xmax_score])
    new_props=numpy.stack(new_props)
    
    col_name=["xmin","xmax","xmin_score","xmax_score"]
    new_df=pandas.DataFrame(new_props,columns=col_name)  
    new_df["score"]=new_df.xmin_score*new_df.xmax_score
    
    new_df=new_df.sort_values(by="score",ascending=False)
    
    video_info=video_dict[video_name]
    video_frame=video_info['duration_frame']
    video_second=video_info['duration_second']
    feature_frame=video_info['feature_frame']
    corrected_second=float(feature_frame)/video_frame*video_second
    
    try:
        gt_xmins=[]
        gt_xmaxs=[]
        for idx in range(len(video_info["annotations"])):
            gt_xmins.append(video_info["annotations"][idx]["segment"][0]/corrected_second)
            gt_xmaxs.append(video_info["annotations"][idx]["segment"][1]/corrected_second)
        new_iou_list=[]
        for j in range(len(new_df)):
            tmp_new_iou=max(iou_with_anchors(new_df.xmin.values[j],new_df.xmax.values[j],gt_xmins,gt_xmaxs))
            new_iou_list.append(tmp_new_iou)
            
        new_ioa_list=[]
        for j in range(len(new_df)):
            tmp_new_ioa=max(ioa_with_anchors(new_df.xmin.values[j],new_df.xmax.values[j],gt_xmins,gt_xmaxs))
            new_ioa_list.append(tmp_new_ioa)
        new_df["match_iou"]=new_iou_list
        new_df["match_ioa"]=new_ioa_list
    except:
        pass
    new_df.to_csv("./output/PGM_proposals/"+video_name+".csv",index=False)
    
parser = argparse.ArgumentParser(description="Boundary Sensitive Network")
parser.add_argument('start_idx', type=int)
parser.add_argument('end_idx', type=int)
args = parser.parse_args()
video_dict= load_json("./data/activitynet_annotations/anet_anno_action.json")
result_dict={}
video_list=video_dict.keys()[args.start_idx:args.end_idx]

for video_name in video_list:
    generateProposals(video_name,video_dict)
