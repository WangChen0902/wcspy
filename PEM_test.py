# -*- coding: utf-8 -*-
"""
Created on Fri Feb 17 20:25:55 2017

@author: wzmsltw
"""

import tensorflow as tf
import numpy as np
import pandas as pd
import PEM_load_data
import sys

def PEM_inference(X,config):
    net=0.1*tf.matmul(X, config.W["iou_0"]) + config.biases["iou_0"]
    net=tf.nn.relu(net)
    net=0.1*tf.matmul(net, config.W["iou_1"]) + config.biases["iou_1"]
    net=tf.nn.sigmoid(net)
    anchors_iou=net
    anchors_iou=tf.reshape(anchors_iou,[-1])
    return anchors_iou
    

class Config(object):
    """
    define a class to store parameters,
    the input should be feature mat of training and testing
    """
    def __init__(self):
        self.batch_size=16
        with tf.variable_scope("latent_net"):
            self.W = {
                'iou_0': tf.Variable(tf.truncated_normal([16, 256])),
                'iou_1': tf.Variable(tf.truncated_normal([256, 1]))}
            self.biases = {
                'iou_0': tf.Variable(tf.truncated_normal([256])),
                'iou_1': tf.Variable(tf.truncated_normal([1]))}


if __name__ == "__main__":
    config = Config()
    
    X_feature = tf.placeholder(tf.float32, [None,16])

    prop_score=PEM_inference(X_feature,config)
    
    model_saver=tf.train.Saver(var_list=tf.trainable_variables(),max_to_keep=80)
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    tf_config.log_device_placement =True
    sess=tf.InteractiveSession(config=tf_config)
    tf.global_variables_initializer().run()  
    model_saver.restore(sess,"models/PEM/pem_model_best")  

    dataSet = sys.argv[1]
    file_path = "./output/TEM_results/"+dataSet+"/"
    video_dict = PEM_load_data.get_simple_file_name(file_path)

    FullDict=PEM_load_data.getTestData(dataSet)

    batch_video_list=PEM_load_data.getBatchList(video_dict,config.batch_size)
    video_list=video_dict
    for idx in range(len(video_list)):
        video_name=video_list[idx]
        prop_dict=FullDict[video_name]
        batch_feature,batch_iou_list,batch_ioa_list=PEM_load_data.prop_dict_data({"data":prop_dict})
        #print(batch_feature)
        if batch_feature==[]:
            continue
        out_score=sess.run(prop_score,feed_dict={X_feature:batch_feature})  
                                                          
        out_score=np.reshape(out_score,[-1])
        out_score_len = len(out_score)
        # print("out_score: ", out_score)
        # print("len out_score: ", len(out_score))
        xmin_list=prop_dict["xmin"][:out_score_len]
        # print("xmin_list: ", xmin_list)
        # print("len xmin_list: ", len(xmin_list))
        xmax_list=prop_dict["xmax"][:out_score_len]
        xmin_score_list=prop_dict["xmin_score"][:out_score_len]
        xmax_score_list=prop_dict["xmax_score"][:out_score_len]
        latentDf=pd.DataFrame()
        latentDf["xmin"]=xmin_list
        latentDf["xmax"]=xmax_list
        latentDf["xmin_score"]=xmin_score_list
        latentDf["xmax_score"]=xmax_score_list
        latentDf["iou_score"]=out_score
        
        
        latentDf.to_csv("./output/PEM_results/"+dataSet+"/"+video_name+".csv",index=False)

