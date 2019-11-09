# -*- coding: utf-8 -*-
from scipy.interpolate import interp1d
import pandas
import argparse
import numpy
import json
import tensorflow as tf
import os
import sys

def load_json(file):
    with open(file) as json_file:
        data = json.load(json_file)
        return data

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
        'startendLabel': tf.FixedLenFeature(shape=(100,), dtype=tf.int64)
    }
    parsed_example = tf.parse_single_example(serialized=example_proto, features=dics)
    parsed_example['tensor'] = tf.decode_raw(parsed_example['tensor'], tf.float32)
    parsed_example['tensor'] = tf.reshape(parsed_example['tensor'], parsed_example['tensor_shape'])
    return parsed_example

def generateFeature(set_name,video_name):

    num_sample_start=4
    num_sample_end=4
    num_sample_action=8
    num_sample_interpld = 3

    adf=pandas.read_csv("./output/TEM_results/"+set_name+"/"+video_name+".csv")
    score_action=adf.action.values[:]
    seg_xmins = adf.xmin.values[:]
    seg_xmaxs = adf.xmax.values[:]
    video_scale = len(adf)
    video_gap = seg_xmaxs[0] - seg_xmins[0]
    video_extend = int(video_scale / 4 + 10)
    pdf=pandas.read_csv("./output/PGM_proposals/"+set_name+"/"+video_name+".csv")
    
    pdf=pdf[:100]
    tmp_zeros=numpy.zeros([video_extend])    
    score_action=numpy.concatenate((tmp_zeros,score_action,tmp_zeros))
    tmp_cell = video_gap
    tmp_x = [-tmp_cell/2-(video_extend-1-ii)*tmp_cell for ii in range(video_extend)] + \
             [tmp_cell/2+ii*tmp_cell for ii in range(video_scale)] + \
              [tmp_cell/2+seg_xmaxs[-1] +ii*tmp_cell for ii in range(video_extend)]
    f_action=interp1d(tmp_x,score_action,axis=0)
    feature_bsp=[]

    for idx in range(len(pdf)):
        xmin=pdf.xmin.values[idx]
        xmax=pdf.xmax.values[idx]
        xlen=xmax-xmin
        xmin_0=xmin-xlen/5
        xmin_1=xmin+xlen/5
        xmax_0=xmax-xlen/5
        xmax_1=xmax+xlen/5
        #start
        plen_start= (xmin_1-xmin_0)/(num_sample_start-1)
        plen_sample = plen_start / num_sample_interpld
        tmp_x_new = [ xmin_0 - plen_start/2 + plen_sample * ii for ii in range(num_sample_start*num_sample_interpld +1 )] 
        tmp_y_new_start_action=f_action(tmp_x_new)
        tmp_y_new_start = [numpy.mean(tmp_y_new_start_action[ii*num_sample_interpld:(ii+1)*num_sample_interpld+1]) for ii in range(num_sample_start) ]
        #end
        plen_end= (xmax_1-xmax_0)/(num_sample_end-1)
        plen_sample = plen_end / num_sample_interpld
        tmp_x_new = [ xmax_0 - plen_end/2 + plen_sample * ii for ii in range(num_sample_end*num_sample_interpld +1 )] 
        tmp_y_new_end_action=f_action(tmp_x_new)
        tmp_y_new_end = [numpy.mean(tmp_y_new_end_action[ii*num_sample_interpld:(ii+1)*num_sample_interpld+1]) for ii in range(num_sample_end) ]
        #action
        plen_action= (xmax-xmin)/(num_sample_action-1)
        plen_sample = plen_action / num_sample_interpld
        tmp_x_new = [ xmin - plen_action/2 + plen_sample * ii for ii in range(num_sample_action*num_sample_interpld +1 )] 
        tmp_y_new_action=f_action(tmp_x_new)
        tmp_y_new_action = [numpy.mean(tmp_y_new_action[ii*num_sample_interpld:(ii+1)*num_sample_interpld+1]) for ii in range(num_sample_action) ]
        tmp_feature = numpy.concatenate([tmp_y_new_action,tmp_y_new_start,tmp_y_new_end])
        feature_bsp.append(tmp_feature)
    feature_bsp = numpy.array(feature_bsp)
    numpy.save("./output/PGM_feature/"+set_name+"/"+video_name,feature_bsp)

# sets = ['train', 'val', 'test']
set_name = sys.argv[1]

PATH = '../wangchen/'+set_name+'/'
test_file = get_file_list(PATH)
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
    while True:
        try:
            startendLabel = sess.run(next_element['startendLabel'])
        except tf.errors.OutOfRangeError:
            print('end!')
            break
        except tf.errors.DataLossError:
            print('end!')
            break
        else:
            video_name = simple_name + '_' + str(i)
            print(video_name)
            generateFeature(set_name,video_name)
        i = i+1