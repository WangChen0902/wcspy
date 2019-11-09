# -*- coding: utf-8 -*-
import cv2
import numpy as np
import h5py
ROOT_PATH="/Users/liwenxu/Public/学习/program/PycharmProjects/footbalData/dataset"
ROOT_PATH="/data/dataset/dataset"
def getVideoAllframe(tagName):
    video_full_path=ROOT_PATH+"/"+tagName.split("_")[0]+"/"+"data_"+tagName+"_3d.mp4"
    biaoji_full_path=ROOT_PATH+"/"+tagName.split("_")[0]+"/"+"biaoji.txt"
    f=open(biaoji_full_path,"r")
    line_count=0
    start=0
    for line in f.readlines():
        if line_count==0:
            line_count+=1
            continue
        if line.split(",")[2].split(" ")[0].split("_")[2]==tagName.split("_")[1]:
            start=int(line.split(",")[2].split(" ")[1])
    if start==0:
        print("exception!   匹配失败")
    cap = cv2.VideoCapture(video_full_path)
    frames_num = cap.get(7)
    return int(frames_num) - int(start)
def flatten(l):
    for k in l:
      if not isinstance(k, (list, tuple)):
        yield k
      else:
        yield from flatten(k)
def getSegmentNumber(segmentCount,exceptions,tagName):
    if segmentCount==0: #第一个片段
        return exceptions[0]-4,0
    elif segmentCount<len(exceptions):
        return exceptions[segmentCount]-exceptions[segmentCount-1]-7,exceptions[segmentCount-1]+3
    else: #最后一个片段
        return getVideoAllframe(tagName)-exceptions[segmentCount-1]-4,exceptions[segmentCount-1]+3
def getSnippetCount(num_of_one_segment):
    # if num_of_one_segment%16==0:
    #     return num_of_one_segment//16
    # else:
    #     return  (num_of_one_segment//16) +1
    return num_of_one_segment//16
def checkIou(intersection,duration):
    if  intersection/duration>0.7:
        return 1
    else:
        return 0
def getSnippetLabel(start,end,groundtruth): #你需要更改一下这个的代码
    for item in groundtruth:
        item_start=int(item.split(" ")[0])
        item_end=int(item.split(" ")[1])
        if item_start>end or start>item_end:
            continue
        else:
            if item_start>start: #左相交
                return checkIou(end-item_start,end-start)  #不允许两个ground truth交叉 所以直接返回应该就ok
            elif end>item_end: #右相交
                return checkIou(item_end-start,end-start)
            else: # snippet被包含
                return 1
    return 0
def changeToint(exceptions):
    result=[]
    for i in exceptions:
        result.append(int(i.split(" ")[0]))
    return result
def getSnippetLabelBytag(tagName):
        f_exception=open(ROOT_PATH+"/"+tagName.split("_")[0]+"/result/"+"data_"+tagName+"_3dexceptiondui.txt")
        f_groundtruth=open(ROOT_PATH+"/"+tagName.split("_")[0]+"/result/"+tagName+"groundtruth.txt")
        groundtruth=f_groundtruth.readlines()
        exceptions=changeToint(f_exception.readlines())
        num_of_segment=len(exceptions)+1
        result=[]
        snippet=[0]
        currentSnippet=0
        for i in range(num_of_segment):#per segment
            num_of_one_segment,boundary=getSegmentNumber(i,exceptions,tagName)
            segmentGroundTruth=[]
            snippetCount=getSnippetCount(num_of_one_segment)
            for a in range(snippetCount): #per snippet
                start=a*16+boundary
                end=(a+1)*16+boundary
                segmentGroundTruth.append(getSnippetLabel(start,end,groundtruth))
            result.append(segmentGroundTruth)
            snippet.append(snippetCount+currentSnippet)
            currentSnippet+=snippetCount
        return result,snippet
result, snippet=getSnippetLabelBytag("19_2")
print('s',snippet)
print('r',len(result))
new = []
for r in result:
    print(len(r))
    for ri in r:
        new.append(ri)
print(len(new))
#for i in result:
#     print('ss',i)
# h5Path = "/data/dataset/program/video_summary/dataset/20_1videoFeature.hdf5"
# f = h5py.File(h5Path, 'r')
# tensor_key = ""
# label_key = ""
# for key in f.keys():
#     if key[:6] == "tensor":
#         tensor_key = key
#     elif key[:5] == "label":
#         label_key = key
# print(np.shape(f[tensor_key][:]))
# print(np.shape(f[label_key][:]))