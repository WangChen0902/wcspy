import tensorflow as tf
import numpy as np
import pandas as pd
import TEM_load_data
import os
import datetime
import sys

tscale = 100
tgap = 1. / tscale

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

def TEM_inference(X_feature,config):
    net=tf.layers.conv1d(inputs=X_feature,filters=512,kernel_size=3,strides=1,padding='same')
    net=tf.layers.batch_normalization(net,training=True)
    net=tf.nn.relu(net)
    net=tf.layers.conv1d(inputs=net,filters=512,kernel_size=3,strides=1,padding='same')
    net=tf.layers.batch_normalization(net,training=True)
    net=tf.nn.relu(net)
    net=0.1*tf.layers.conv1d(inputs=net,filters=4,kernel_size=1,strides=1,padding='same')
    net=tf.nn.softmax(net)

    anchors_action=net[:,:,0]
    anchors_start=net[:,:,1]
    anchors_end=net[:,:,2]
    anchors_background=net[:,:,3]
    
    scores={"anchors_action":anchors_action,
            "anchors_start":anchors_start,
            "anchors_end":anchors_end}
    return scores
    
def getMyProposalDataTest(my_len):
    """Load data during testing
    """
    batch_anchor_xmin=[]
    batch_anchor_xmax=[]
    for i in range(my_len):
        tmp_anchor_xmin=[tgap*i for i in range(tscale)]
        tmp_anchor_xmax=[tgap*i for i in range(1,tscale+1)]
        batch_anchor_xmin.append(list(tmp_anchor_xmin))    
        batch_anchor_xmax.append(list(tmp_anchor_xmax)) 
    batch_anchor_xmin=np.array(batch_anchor_xmin)
    batch_anchor_xmax=np.array(batch_anchor_xmax)
    return batch_anchor_xmin,batch_anchor_xmax

class Config(object):
    def __init__(self):
        #common information
        self.learning_rates=[0.005]*10+[0.001]*10
        self.training_epochs = len(self.learning_rates)
        self.n_inputs =  2048
        self.batch_size = 1
        self.input_steps = 100

if __name__ == "__main__":
    # sets = ['train', 'val', 'test']
    set_name = sys.argv[1]

    PATH = '../wangchen/'+set_name+'/'
    test_file = get_file_list(PATH)
    options = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.ZLIB)
    
    config = Config()
    test_dataset = tf.data.TFRecordDataset(test_file, compression_type='ZLIB')
    new_test_dataset = test_dataset.map(parse_function)
    new_test_dataset = new_test_dataset.apply(tf.contrib.data.batch_and_drop_remainder(config.batch_size))
    iter = tf.data.Iterator.from_structure(new_test_dataset.output_types)
    init_test_op = iter.make_initializer(new_test_dataset)
    next_element = iter.get_next()

    X_feature = next_element['tensor']
    X_feature = tf.cast(X_feature, dtype=tf.float32)
    X_feature.set_shape([config.batch_size,config.input_steps,config.n_inputs])

    tem_scores=TEM_inference(X_feature,config)
    
    print('This is trainable variables():  ', tf.trainable_variables())
    for var in tf.trainable_variables():
        print('name: ', var.name)

    model_saver=tf.train.Saver(var_list=tf.trainable_variables(),max_to_keep=80)
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    tf_config.log_device_placement =True
    sess=tf.InteractiveSession(config=tf_config)
    tf.global_variables_initializer().run()  
    model_saver.restore(sess,"models/TEM/tem_model_checkpoint")  

    batch_result_action=[]
    batch_result_start=[]
    batch_result_end=[]
    batch_result_xmin=[]
    batch_result_xmax=[]
    
    list_len = 0
    sess.run(init_test_op)
    while True:
        try:
            out_scores=sess.run(tem_scores)
            # print('OUT_SCORES: ', out_scores)
        except tf.errors.OutOfRangeError:
            print('end!')
            break
        except tf.errors.DataLossError:
            print('end!')
            break
        else:
            list_len = list_len + 1
            batch_result_action.append(out_scores["anchors_action"])
            batch_result_start.append(out_scores["anchors_start"])
            batch_result_end.append(out_scores["anchors_end"])
            batch_anchor_xmin,batch_anchor_xmax=getMyProposalDataTest(len(out_scores["anchors_action"]))
            batch_result_xmin.append(batch_anchor_xmin)
            batch_result_xmax.append(batch_anchor_xmax)

    columns=["action","start","end","xmin","xmax"]
    print(len(out_scores["anchors_action"]))
    print('action:',len(batch_result_action))
    print('start:',len(batch_result_start))
    print('end:',len(batch_result_end))
    print('xmin:',len(batch_result_xmin))
    print('xmax:',len(batch_result_xmax))
    print('videolist:',len(test_file))
    # print(test_file)
    print(list_len)

    snip_len_dict = {}
    snip_len_path = set_name+'_snip_count.txt'
    f = open(snip_len_path, 'r')
    line = f.readline()
    while(line):
        snip_len_dict[line.split(' ')[0]] = int(line.split(' ')[1])
        # snip_len.append(int(line.split(' ')[1]))
        # simple_name_list.append(line.split(' ')[0])
        line = f.readline()
    # print(snip_len)
    
    simple_name_list = []
    for idx_test in test_file:
        simple_name = idx_test.split('/')[-1]
        simple_name = simple_name.split('.')[0]
        simple_name = simple_name.split('v')[0]
        simple_name_list.append(simple_name)
    
    print(simple_name_list)

    i = 0
    k = -1
    sum = 0
    label = 0
    for idx in range(list_len):
        b_action=batch_result_action[idx]
        b_start=batch_result_start[idx]
        b_end=batch_result_end[idx]
        b_xmin=batch_result_xmin[idx]
        b_xmax=batch_result_xmax[idx]
        # print('--------')
        # print(len(b_action))
        # print(len(b_xmin))
        # print(len(b_xmax))
        # print('--------')
        for j in range(len(b_action)):
            if i == sum:
                k = k + 1
                print(i, sum)
                label = 0
                print(k)
                # print(len(snip_len))
                sum = sum + snip_len_dict[simple_name_list[k]]

            # print(str(simple_name_list[k]) + '_' + str(label))
            # print(len(b_action))
            tmp_action=b_action[j]
            tmp_start=b_start[j]
            tmp_end=b_end[j]
            tmp_xmin=b_xmin[j]
            tmp_xmax=b_xmax[j]
            tmp_result=np.stack((tmp_action,tmp_start,tmp_end,tmp_xmin,tmp_xmax),axis=1)
            tmp_df=pd.DataFrame(tmp_result,columns=columns)  
            tmp_df.to_csv("./output/TEM_results/" + set_name + "/" + str(simple_name_list[k]) + "_" + str(label) +".csv",index=False)
            i = i + 1
            label = label + 1