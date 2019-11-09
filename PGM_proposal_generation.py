# -*- coding: utf-8 -*-
import json
import numpy
import pandas
import argparse
import tensorflow as tf
import os
import sys

def get_file_list(dir):
    files = os.listdir(dir)
    full_names = []
    for file in files:
        file = dir + file
        full_names.append(file)
    return full_names

def parse_function(example_proto):
    dics = {
        'tensor': tf.FixedLenFeature(shape=(), dtype=tf.string),
        'tensor_shape': tf.FixedLenFeature(shape=(2,), dtype=tf.int64),
        'actionLabel': tf.FixedLenFeature(shape=(100,),dtype=tf.int64),
        'startLabel': tf.FixedLenFeature(shape=(100,), dtype=tf.int64),
        'endLabel': tf.FixedLenFeature(shape=(100,), dtype=tf.int64)
    }
    parsed_example = tf.parse_single_example(serialized=example_proto, features=dics)
    parsed_example['tensor'] = tf.decode_raw(parsed_example['tensor'], tf.float32)
    parsed_example['tensor'] = tf.reshape(parsed_example['tensor'], parsed_example['tensor_shape'])
    return parsed_example

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
    
    # print('xmin: ', int_xmin)
    # print('xmax: ', int_xmax)
    inter_len = numpy.maximum(int_xmax - int_xmin, 0.)
    # print('inter_len: ', inter_len)
    # print('len_anchors: ', len_anchors)
    union_len = len_anchors - inter_len + box_max - box_min
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

def generateProposals(set_name,video_name,actionLabel):
    #snip_len = 0
    #snip_len_path = set_name+'_snip_count.txt'
    #f = open(snip_len_path, 'r')
    #line = f.readline()
    #while(line):
    #    if line.split(' ')[0]==video_name:
    #        snip_len = int(line.split(' ')[1])
    #        break
    #    line = f.readline()
    #print(snip_len)

    tscale = 100
    tgap = 1./tscale
    peak_thres=0.5

    gt_xmins=[]
    gt_xmaxs=[]
    idx = 0
    # print(actionLabel)
    while idx < len(actionLabel):
        if actionLabel[idx]==1:
            tmp_gt_min = numpy.maximum(idx - 1, 0.)
            gt_xmins.append(float(tmp_gt_min)/float(tscale))
            start_idx = idx
            # print('1-idx: ', idx)
            for idx2 in range(start_idx, len(actionLabel)):
                if actionLabel[idx2]==0 or idx2==len(actionLabel)-1:
                    gt_xmaxs.append(float(idx2)/float(tscale))
                    # print('2-idx: ', idx2)
                    idx = idx2 + 1
                    break
        else:
            idx = idx + 1
    if gt_xmins==[]:
        gt_xmins.append(0)
    if gt_xmaxs==[]:
        gt_xmaxs.append(0)
    # print(gt_xmins)
    # print(gt_xmaxs)

    gt_len = []
    for i in range(len(gt_xmaxs)):
        tmp_len = gt_xmaxs[i] - gt_xmins[i]
        gt_len.append(tmp_len)
    
    max_gt_len = numpy.max(numpy.array(gt_len)) + 0.03
    min_gt_len = 0.03
    #min_gt_len = numpy.min(numpy.array(gt_len))
    # print(max_gt_len)
    # print(min_gt_len)

    #startend_scores=[]
    #for k in range(snip_len):
    #tdf=pandas.read_csv("./output/TEM_results/"+set_name+"/"+video_name+"_"+str(k)+".csv")
    tdf=pandas.read_csv("./output/TEM_results/"+set_name+"/"+video_name+".csv")
    start_scores = tdf.start.values[:]
    end_scores = tdf.end.values[:]
    #    startend_scores.extend(tmp_score)
    
    start_bins=numpy.zeros(len(start_scores))
    end_bins=numpy.zeros(len(end_scores))
    start_bins[0]=1
    end_bins[-1]=1
    for idx in range(1,tscale-1):
        # if startend_scores[idx]>startend_scores[idx+1] and startend_scores[idx]>startend_scores[idx-1]:
        #     startend_bins[idx]=1
        if start_scores[idx]>0.1:
            start_bins[idx]=1

    for idx in range(1,tscale-1):
        if end_scores[idx]>0.1:
            end_bins[idx]=1

    xmin_list=[]
    xmin_idx_list=[]
    xmin_score_list=[]
    xmax_list=[]
    xmax_idx_list=[]
    xmax_score_list=[]

    for j in range(tscale):
        if start_bins[j]==1:
            xmin_list.append(tgap/2+tgap*j)
            xmin_idx_list.append(j)
            xmin_score_list.append(start_scores[j])
        if end_bins[j]==1:
            xmax_list.append(tgap/2+tgap*j)
            xmax_idx_list.append(j)
            xmax_score_list.append(end_scores[j])
            
    new_props=[]
    for ii in range(len(xmax_list)):
        tmp_xmax_idx=xmax_idx_list[ii]
        tmp_xmax=xmax_list[ii]
        tmp_xmax_score=xmax_score_list[ii]
        
        for ij in range(len(xmin_list)):
            diff = abs(ii-ij)
            tmp_xmin_idx=xmin_idx_list[ij]
            tmp_xmax_idx=tmp_xmin_idx+4
            tmp_xmin=xmin_list[ij]
            tmp_xmin_score=xmin_score_list[ij]
            if tmp_xmin>=tmp_xmax:
                break
            tmp_xlen = tmp_xmax - tmp_xmin
            if tmp_xlen>=min_gt_len and tmp_xlen<=max_gt_len:
                new_props.append([tmp_xmin,tmp_xmax,tmp_xmin_score,tmp_xmax_score])

    if new_props==[]:
        print(video_name)
    if new_props!=[]:
        new_props=numpy.stack(new_props)
    
    col_name=["xmin","xmax","xmin_score","xmax_score"]
    new_df=pandas.DataFrame(new_props,columns=col_name)  
    new_df["score"]=new_df.xmin_score*new_df.xmax_score
    
    new_df=new_df.sort_values(by="score",ascending=False)
    
    # try:
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
    # except:
    #     pass
    new_df.to_csv("./output/PGM_proposals/"+set_name+"/"+video_name+".csv",index=False)

# sets = ['train', 'val', 'test']
set_name = sys.argv[1]

PATH = '../wangchen/'+set_name+'/'
test_file = get_file_list(PATH)
# test_file = ['D:/wangchen/val/19_2videoFeature.tfrecord']
options = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.ZLIB)

# "./output/TEM_results/"+video_name+".csv"
for idx_test in test_file:
    simple_name = idx_test.split('/')[-1]
    simple_name = simple_name.split('.')[0]
    simple_name = simple_name.split('v')[0]
    
    single_file = [idx_test]
    test_dataset = tf.data.TFRecordDataset(single_file, compression_type='ZLIB')
    new_test_dataset = test_dataset.map(parse_function)
    iter = tf.data.Iterator.from_structure(new_test_dataset.output_types)
    init_test_op = iter.make_initializer(new_test_dataset)
    next_element = iter.get_next()
    sess = tf.InteractiveSession()
    sess.run(init_test_op)

    i=0
    full_action_label = []
    print(simple_name)
    while True:
        try:
            actionLabel = sess.run(next_element['actionLabel'])
            full_action_label.extend(actionLabel)
        except tf.errors.OutOfRangeError:
            print('end!')
            break
        except tf.errors.DataLossError:
            print('end!')
            break
        else:
            video_name = simple_name + '_' + str(i)
            print(video_name)
            generateProposals(set_name,video_name,actionLabel)
        i = i+1
    #generateProposals(set_name,simple_name,full_action_label)
