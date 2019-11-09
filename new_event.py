import numpy as np
import pandas as pd
import json
import os
import sys
import operator

def get_simple_file_name(dir):
    files = os.listdir(dir)
    full_names = []
    for file in files:
        # file = dir + file
        file = file.split('.')[0]
        full_names.append(file)
    return full_names

def get_full_file_name(dir):
    files = os.listdir(dir)
    full_names = []
    for file in files:
        file = file.split('_')[0]+'_'+file.split('_')[1]
        if file not in full_names:
            full_names.append(file)
    return full_names

def load_json(file):
    with open(file) as json_file:
        data = json.load(json_file)
        return data

def get_final(dataset, index):
    final = []
    types = []
    first_name = index.split('_')[0]
    file_path = '/data/dataset/dataset/'+first_name+'/result/'+index+'final.txt'
    final_txt = open(file_path, 'r', encoding='UTF-8')
    line = final_txt.readline()
    while line:
        line_list = line.split(' ')
        start = int(line_list[1])
        end = start + int(line_list[2])
        event_type = int(line_list[4])
        if event_type==0 or event_type==1 or event_type==2:
            event = [start, end]
            final.append(event)
            types.append(event_type)
        line = final_txt.readline()
    return final,types

def get_recall(groundtruth,types, proposal,index):
    TP = 0
    count = len(groundtruth)
    proposal_len = len(proposal)
    for g_index in range(len(groundtruth)):
        g =groundtruth[g_index]
        flag = False
        for p in proposal:
            if g[0]>p[0]:
                start0 = p[0]
                start1 = g[0]
            else:
                start0 = g[0]
                start1 = p[0]
            if g[1]>p[1]:
                end0 = p[1]
                end1 = g[1]
            else:
                end0 = g[1]
                end1 = p[1]
            inner_len = end0 - start1
            outer_len = end1 - start0
            iou = float(inner_len) / float(outer_len)
            if iou>=0.5:
                TP = TP +1
                flag = True
                break
    recall = TP/count
    print('count: ',count)
    print('proposal_len: ',proposal_len)
    print('TP: ',TP)
    return recall

path_event = '/data/dataset/program/snippet_feature_pretrain_kinetic/event_h5/Eventframes/'
path_96 = '/data/dataset/program/snippet_feature_pretrain_kinetic/event_h5_96snippet/Eventframes/'
path_112 = '/data/dataset/program/snippet_feature_pretrain_kinetic/event_h5_112snippet/Eventframes/' 
path_16 = '/data/dataset/program/snippet_feature_pretrain_kinetic/event_h5_16snippet/Eventframes/'

dataSet = sys.argv[1]
file_path = "./output/TEM_results/"+dataSet+"/"
video_list = get_simple_file_name(file_path)
full_video_list = get_full_file_name(file_path)
# print(full_video_list)
json_file = load_json("./output/"+dataSet+"_result_proposal.json")

results = json_file['results']

info_list = {}
for info in results:
    video_name_list = info.split('_')
    first_name = video_name_list[0]+'_'+video_name_list[1]
    second_name = video_name_list[2]
    if first_name not in info_list:
        info_list[first_name] = {}
    info_list[first_name][int(second_name)] = results[info]
new_info_list = {}
for index in info_list:
    sorted_info = sorted(info_list[index].items(), key=lambda x:x[0])
    new_info_list[index] = sorted_info
#print(new_info_list['4_2'])
# sorted_info = sorted(info_list.items(), key=lambda x:x[0])
frame_info_list = {}
for i in new_info_list:
    frame_info_list[i] = []
    for j in new_info_list[i]:
        #print(j[0])
        current_index = j[0]
        #print(j[1])
        for k in j[1]:
            new_frame = [int((k['segment'][0]+current_index)*100),int((k['segment'][1]+current_index)*100)]
            frame_info_list[i].append(new_frame)
        # print(j)
for ii in frame_info_list:
    frame_info_list[ii].sort(key=lambda x:x[0])

first_frame = {}
for index in frame_info_list:
    first_frame[index] = []
    for i in range(len(frame_info_list[index])):
        tmp_list = [int(frame_info_list[index][i][0]), int(frame_info_list[index][i][1])]
        first_frame[index].append(tmp_list)

# print(first_frame)

frames_16 = {}

for root, dirs, files in os.walk(path_16):
    for file in files:
        short_name = file.split('.')[0]
        frames_16[short_name] = []
        file_name = os.path.join(path_16, file)
        f = open(file_name, 'r')
        line = f.readline()
        while line:
            info_list = line.split(' ')
            # print(info_list)
            frames_16[short_name].append([int(info_list[2]), int(info_list[3])])
            line = f.readline()

for fir in first_frame:
    event = first_frame[fir]
    more = []
    for i in range(len(event)-1):
        if i==0:
            if event[i][0]>=3:
                s = 0
                e = event[i][0]
                l = s+3
                while l<=e:
                    if e-l>=3:
                        more.append([s,l])
                        s = l
                        l = s+3
                    else:
                        more.append([s,e])
                        break
        if event[i+1][0] - event[i][1] >= 3:
            s = event[i][1]
            e = event[i+1][0]
            l = s+3
            while l<=e:
                if e-l>=3:
                    more.append([s,l])
                    s = l
                    l = s+3
                else:
                    more.append([s,e])
                    break
    for m in more:
        event.append(m)
    event.sort()
    print(more)
    first_frame[fir] = event

# print(first_frame['19_2'])
frame_info_list = {}
for index in first_frame:
    if index=='23_2' or index=='29_1':
        continue
    frame_info_list[index] = []
    for i in range(len(first_frame[index])):
        s = int(first_frame[index][i][0])
        e = int(first_frame[index][i][1])
        s_frame = frames_16[index][s][0]
        e_frame = frames_16[index][e][1]
        frame_info_list[index].append([s_frame, e_frame])

for index in frame_info_list:
    if index=='23_2' or index=='29_1':
        continue
    f = open(path_event+dataSet+'/'+str(index)+'.txt', 'w')
    for iii in range(len(first_frame[index])):
        print_str = str(first_frame[index][iii][0])+' '+str(first_frame[index][iii][1])+' '+str(frame_info_list[index][iii][0])+' '+str(frame_info_list[index][iii][1])
        f.write(print_str+'\n')

first_frame_96 = {}
first_frame_112 = {}
out_snip_count = './out_count.txt'
out_snip_file = open(out_snip_count, 'r')
line = out_snip_file.readline()
while line:
    line_li = line.split(' ')
    first_frame_112[line_li[0]] = []
    first_frame_96[line_li[0]] = []
    for ij in range(int(line_li[1])):
        if ij%6==0 and ij+5<int(line_li[1]):
            tmp = [ij, ij+5]
            first_frame_96[line_li[0]].append(tmp)
        if ij%7==0 and ij+6<int(line_li[1]):
            tmp = [ij, ij+6]
            first_frame_112[line_li[0]].append(tmp)
    line = out_snip_file.readline()

frame_info_list = {}
for index in first_frame_96:
    if index=='23_2' or index=='29_1':
        continue
    frame_info_list[index] = []
    for i in range(len(first_frame_96[index])):
        s = int(first_frame_96[index][i][0])
        e = int(first_frame_96[index][i][1])
        s_frame = frames_16[index][s][0]
        e_frame = frames_16[index][e][1]
        frame_info_list[index].append([s_frame, e_frame])

for index in frame_info_list:
    if index=='23_2' or index=='29_1':
        continue
    f = open(path_96+str(index)+'.txt', 'w')
    for iii in range(len(first_frame_96[index])):
        print_str = str(first_frame_96[index][iii][0])+' '+str(first_frame_96[index][iii][1])+' '+str(frame_info_list[index][iii][0])+' '+str(frame_info_list[index][iii][1])
        f.write(print_str+'\n')

frame_info_list = {}
for index in first_frame_112:
    if index=='23_2' or index=='29_1':
        continue
    frame_info_list[index] = []
    for i in range(len(first_frame_112[index])):
        s = int(first_frame_112[index][i][0])
        e = int(first_frame_112[index][i][1])
        s_frame = frames_16[index][s][0]
        e_frame = frames_16[index][e][1]
        frame_info_list[index].append([s_frame, e_frame])

for index in frame_info_list:
    if index=='23_2' or index=='29_1':
        continue
    f = open(path_112+str(index)+'.txt', 'w')
    for iii in range(len(first_frame_112[index])):
        print_str = str(first_frame_112[index][iii][0])+' '+str(first_frame_112[index][iii][1])+' '+str(frame_info_list[index][iii][0])+' '+str(frame_info_list[index][iii][1])
        f.write(print_str+'\n')

new_frame_info_list = {}
for index in frame_info_list:
    if index=='23_2' or index=='29_1':
        continue
    new_frame_info_list[index] = []
    for i in range(len(frame_info_list[index])):
        s = int(frame_info_list[index][i][0])+8
        e = int(frame_info_list[index][i][1])-8
        new_frame_info_list[index].append([s, e])

total_recall = 0
count = len(new_frame_info_list)
for index in new_frame_info_list:
    if index=='23_2' or index=='29_1':
        continue
    final,types = get_final(dataSet, index)
    recall = get_recall(final,types,new_frame_info_list[index],index)
    print(recall)
    total_recall = total_recall + recall
print('average: ', total_recall/count)