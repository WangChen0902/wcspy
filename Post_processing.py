# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import json
import os
import sys

def load_json(file):
    with open(file) as json_file:
        data = json.load(json_file)
        return data

def getDatasetDict():
    df=pd.read_csv("./data/activitynet_annotations/video_info_new.csv")
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

def IOU(s1,e1,s2,e2):
    if (s2>e1) or (s1>e2):
        return 0
    Aor=max(e1,e2)-min(s1,s2)
    Aand=min(e1,e2)-max(s1,s2)
    return float(Aand)/Aor

def NMS(df,nms_threshold):
    df=df.sort_values(by="score",ascending=False)
    
    tstart=list(df.xmin.values[:])
    tend=list(df.xmax.values[:])
    tscore=list(df.score.values[:])
    rstart=[]
    rend=[]
    rscore=[]
    while len(tstart)>1 and len(rscore)<101:
        idx=1
        while idx<len(tstart):
            if IOU(tstart[0],tend[0],tstart[idx],tend[idx])>nms_threshold:
                tstart.pop(idx)
                tend.pop(idx)
                tscore.pop(idx)
            else:
                idx+=1
        rstart.append(tstart[0])
        rend.append(tend[0])
        rscore.append(tscore[0])
        tstart.pop(0)
        tend.pop(0)
        tscore.pop(0)
    newDf=pd.DataFrame()
    newDf['score']=rscore
    newDf['xmin']=rstart
    newDf['xmax']=rend
    return newDf

def Soft_NMS(df):
    df=df.sort_values(by="score",ascending=False)
    
    tstart=list(df.xmin.values[:])
    tend=list(df.xmax.values[:])
    tscore=list(df.score.values[:])
    
    rstart=[]
    rend=[]
    rscore=[]

    while len(tscore)>1 and len(rscore)<101:
        max_index=tscore.index(max(tscore))
        for idx in range(0,len(tscore)):
            if idx!=max_index:
                tmp_iou=IOU(tstart[max_index],tend[max_index],tstart[idx],tend[idx])
                tmp_width=tend[max_index]-tstart[max_index]
                if tmp_iou>0.65+0.25*tmp_width:#*1/(1+np.exp(-max_index)):
                    
                    tscore[idx]=tscore[idx]*np.exp(-np.square(tmp_iou)/0.75)
            
        rstart.append(tstart[max_index])
        rend.append(tend[max_index])
        rscore.append(tscore[max_index])
        tstart.pop(max_index)
        tend.pop(max_index)
        tscore.pop(max_index)
                
    newDf=pd.DataFrame()
    newDf['score']=rscore
    newDf['xmin']=rstart
    newDf['xmax']=rend
    return newDf


def min_max(x):
    x=(x-min(x))/(max(x)-min(x))
    return x

def get_simple_file_name(dir):
    files = os.listdir(dir)
    full_names = []
    for file in files:
        # file = dir + file
        file = file.split('.')[0]
        full_names.append(file)
    return full_names


dataSet = sys.argv[1]
file_path = "./output/TEM_results/"+dataSet+"/"
video_dict = get_simple_file_name(file_path)
video_list = video_dict
result_dict={}

# COMBINE = False

for i in range(len(video_list)):
    video_name=video_list[i]

    df=pd.read_csv("./output/PEM_results/"+dataSet+"/"+video_name+".csv")

    df['score']=df.iou_score.values[:]*df.xmin_score.values[:]*df.xmax_score.values[:]
    if len(df)>1:
        df=NMS(df,0.1)
        #df=Soft_NMS(df)
    
    df=df.sort_values(by="score",ascending=False)
    # video_info=val_dict[video_name]
    # video_duration=float(video_info["duration_frame"]/16*16)/video_info["duration_frame"]*video_info["duration_second"]
    # print(video_duration, video_info["duration_second"])
    proposal_list=[]

    for j in range(len(df)):
        tmp_proposal={}
        tmp_proposal["score"]=df.score.values[j]
        tmp_proposal["segment"]=[max(0,df.xmin.values[j]),min(1,df.xmax.values[j])]
        print(video_name)
        # print(tmp_proposal["segment"])
        #if tmp_proposal["score"]>0.001:
        proposal_list.append(tmp_proposal)
    result_dict[video_name]=proposal_list

output_dict={"version":"VERSION 1.3","results":result_dict,"external_data":{}}
outfile=open("./output/"+dataSet+"_result_proposal.json","w")
json.dump(output_dict,outfile)
outfile.close()

