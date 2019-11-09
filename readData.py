import tensorflow as tf
import os
import numpy as np
import pandas as pd
import sys
# from sklearn import preprocessing

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
        'actionLabel': tf.FixedLenFeature(shape=(100),dtype=tf.int64),
        'startLabel': tf.FixedLenFeature(shape=(100), dtype=tf.int64),
        'endLabel': tf.FixedLenFeature(shape=(100), dtype=tf.int64)
    }
    parsed_example = tf.parse_single_example(serialized=example_proto, features=dics)
    parsed_example['tensor'] = tf.decode_raw(parsed_example['tensor'], tf.float32)
    parsed_example['tensor'] = tf.reshape(parsed_example['tensor'], parsed_example['tensor_shape'])
    return parsed_example


class Config(object):
    def __init__(self):
        #common informationw
        self.learning_rates=[0.005]*10+[0.001]*10
        self.training_epochs = len(self.learning_rates)
        self.n_inputs =  2048
        self.batch_size = 1
        self.input_steps = 100

set_name = sys.argv[1]

TEST_PATH = '/data/dataset/program/wangchen/'+set_name+'/'
# test_file = [TEST_PATH]
test_file = get_file_list(TEST_PATH)
# test_file = ['D:/wangchen/test/46_1videoFeature.tfrecord']
options = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.ZLIB)

out_file = set_name+'_snip_count.txt'
f = open(out_file, 'w')
# set_name = "val"
columns= ["actionLabel", "startLabel", "endLabel"]
for idx in test_file:
    # print(idx)
    # f.write(idx+'\n')
    simple_name = idx.split('/')[-1]
    simple_name = simple_name.split('.')[0]
    simple_name = simple_name.split('v')[0]
    single_file = [idx]
    # config = Config()
    test_dataset = tf.data.TFRecordDataset(single_file, compression_type='ZLIB')
    new_test_dataset = test_dataset.map(parse_function)
    # new_test_dataset = new_test_dataset.apply(tf.contrib.data.batch_and_drop_remainder(config.batch_size))
    iter = tf.data.Iterator.from_structure(new_test_dataset.output_types)
    init_test_op = iter.make_initializer(new_test_dataset)
    next_element = iter.get_next()
    
    sess = tf.InteractiveSession()
    sess.run(init_test_op)
    i=0
    while True:
        try:
            tensor, actionLabel, startLabel, endLabel = sess.run([next_element['tensor'], next_element['actionLabel'], next_element['startLabel'], next_element['endLabel']])
        except tf.errors.OutOfRangeError:
            print('end!')
            break
        except tf.errors.DataLossError:
            print('end!')
            break
        else:
            print('============ example %s ============' %i)
            # print('tensor: shape: %s | type: %s' %(tensor.shape, tensor.dtype))
            print('actionLabel: %s' %(actionLabel))
            print('startLabel: %s' %(startLabel))
            print('endLabel: %s' %(endLabel))
            tmp_result=np.stack((actionLabel,startLabel,endLabel),axis=1)
            tmp_df=pd.DataFrame(tmp_result,columns=columns)
            tmp_df.to_csv('./label_csv/'+set_name+'/'+str(simple_name)+'_'+str(i)+'.csv')
            # f.write('============ example %s ============\n' %i)
            # f.write('actionLabel: %s\n' %(actionLabel))
            # f.write('startendLabel: %s\n' %(startendLabel))
        i = i+1
    f.write(simple_name+' '+str(i)+'\n')