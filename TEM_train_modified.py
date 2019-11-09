# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import TEM_load_data
import os
import datetime
from sklearn import preprocessing

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

def binary_logistic_loss(gt_scores,pred_anchors):
    """Calculate weighted binary logistic loss 
    """ 
    gt_scores = tf.reshape(gt_scores,[-1])
    pred_anchors = tf.reshape(pred_anchors,[-1])
    
    pmask = tf.cast(gt_scores>0.5,dtype=tf.float32)
    num_positive = tf.reduce_sum(pmask)
    num_entries = tf.cast(tf.shape(gt_scores)[0],dtype=tf.float32)    
    ratio = num_entries/num_positive
    coef_0 = 0.5*(ratio)/(ratio-1)
    coef_1 = coef_0*(ratio-1)
    loss = (gt_scores*tf.log(tf.clip_by_value(pred_anchors,1e-8,tf.reduce_max(pred_anchors))))
    # loss = coef_1*pmask*tf.log(pred_anchors) + coef_0*(1.0-pmask)*tf.log(1.0-pred_anchors)
    loss = -tf.reduce_mean(loss)
    num_sample = [tf.reduce_sum(pmask),ratio] 
    return loss

def TEM_loss(anchors,Y_feature,config):
    # print('anchors: ', anchors)
    # print('Y_feature: ', Y_feature)
    losses=binary_logistic_loss(Y_feature,anchors)
    loss = {"loss":losses}
    return loss

def TEM_Train(X_feature,Y_feature,LR,istrain,config):
    """ Model and loss function of temporal evaluation module
    """ 
    # fw = open('debug.txt', 'w+')
    net=tf.layers.conv1d(inputs=X_feature,filters=512,kernel_size=3,strides=1,padding='same')
    net=tf.layers.batch_normalization(net,training=istrain)
    net=tf.nn.relu(net)
    net=tf.nn.dropout(net, keep_prob=0.5)
    # fw.write('layer1: '+str(net)+'\n')
    net=tf.layers.conv1d(inputs=net,filters=512,kernel_size=3,strides=1,padding='same')
    net=tf.layers.batch_normalization(net,training=istrain)
    net=tf.nn.relu(net)
    net=tf.nn.dropout(net, keep_prob=0.5)
    # fw.write('layer2: '+str(net)+'\n')
    net=0.1*tf.layers.conv1d(inputs=net,filters=4,kernel_size=1,strides=1,padding='same')
    net=tf.nn.softmax(net)
    # fw.write('layer3: '+str(net)+'\n')

    anchors_action = net[:,:,0]
    # print("anchors_action: ", anchors_action)
    anchors_start = net[:,:,1]
    anchors_end = net[:,:,2]
    anchors_background = net[:,:,3]
    anchors = tf.stack([anchors_action, anchors_start, anchors_end, anchors_background])
    # net = tf.reshape(net, [config.batch_size, config.input_steps*3])
    # print(net)
    loss=TEM_loss(net,Y_feature,config)

    TEM_trainable_variables=tf.trainable_variables()
    # l2 = 0.001 * sum(tf.nn.l2_loss(tf_var) for tf_var in TEM_trainable_variables)
    # cost = loss["loss_action"]+loss["loss_startend"]+l2
    # loss['l2'] = l2
    # loss['cost'] = cost
    # optimizer=tf.train.AdamOptimizer(learning_rate=LR).minimize(cost,var_list=TEM_trainable_variables)
    opt = tf.train.AdamOptimizer(learning_rate=LR)
    grads = opt.compute_gradients(loss["loss"], var_list=TEM_trainable_variables)
    gs = []
    for i, (g, v) in enumerate(grads):
    	if g is not None:
    		grads[i] = (tf.clip_by_norm(g, 15), v)
    		gs.append(g)
    optimizer = opt.apply_gradients(grads)
    return optimizer,loss,TEM_trainable_variables,Y_feature
    
class Config(object):
    def __init__(self):
        self.input_steps=256
        self.learning_rates=[0.003]*150
        self.training_epochs = len(self.learning_rates)
        self.n_inputs =  2048
        self.batch_size = 1
        self.input_steps=100

if __name__ == "__main__":
    """ define the input and the network""" 
    TRAIN_PATH = '/data/dataset/program/wangchen/train_long/'
    VAL_PATH = '/data/dataset/program/wangchen/val_long/'
    train_file = get_file_list(TRAIN_PATH)
    val_file = get_file_list(VAL_PATH)
    options = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.ZLIB)

    config = Config()
    train_dataset = tf.data.TFRecordDataset(train_file, compression_type='ZLIB')
    new_train_dataset = train_dataset.map(parse_function)
    new_train_dataset = new_train_dataset.apply(tf.contrib.data.batch_and_drop_remainder(config.batch_size))
    val_dataset = tf.data.TFRecordDataset(val_file, compression_type='ZLIB')
    new_val_dataset = val_dataset.map(parse_function)
    new_val_dataset = new_val_dataset.batch(config.batch_size)
    # new_dataset = new_dataset.repeat(config.training_epochs)
    # iterator = new_dataset.make_initializable_iterator()
    # next_element = iterator.get_next()
    iter = tf.data.Iterator.from_structure(new_train_dataset.output_types)
    init_train_op = iter.make_initializer(new_train_dataset)
    init_val_op = iter.make_initializer(new_val_dataset)
    next_element = iter.get_next()
    # fw = open('debug.txt', 'w+')
    X_feature = next_element['tensor']
    enc = preprocessing.OneHotEncoder()
    X_feature = tf.cast(X_feature, dtype=tf.float32)
    X_feature.set_shape([config.batch_size,config.input_steps,config.n_inputs])
    Y_action = next_element['actionLabel']
    Y_action = tf.cast(Y_action, dtype=tf.float32)
    Y_action.set_shape([config.batch_size,config.input_steps])
    Y_start = next_element['startLabel']
    Y_start = tf.cast(Y_start, dtype=tf.float32)
    Y_start.set_shape([config.batch_size,config.input_steps])
    Y_end = next_element['endLabel']
    Y_end = tf.cast(Y_end, dtype=tf.float32)
    Y_end.set_shape([config.batch_size,config.input_steps])
    one_tensor = tf.ones([config.batch_size,config.input_steps], dtype=tf.float32)
    # print("one_tensor: ", one_tensor)
    # print("Y_action: ", Y_action)
    # print("Y_startend: ", Y_startend)
    # fw.write(str(one_tensor)+'\n')
    # fw.write(str(Y_action)+'\n')
    # fw.write(str(Y_startend)+'\n')
    Y_background = tf.subtract(one_tensor,Y_start)
    Y_background = tf.subtract(one_tensor,Y_end)
    Y_background = tf.subtract(Y_background,Y_action)
    Y_feature = tf.stack([Y_action, Y_start, Y_end, Y_background])
    Y_feature = tf.transpose(Y_feature, perm=[1,2,0])
    # Y_feature = tf.reshape(Y_feature, [config.batch_size,config.input_steps*3])
    # print(Y_feature)
    # print(X_feature,Y_action,Y_startend)
    LR = tf.placeholder(tf.float32)
    istrain = tf.placeholder(tf.bool)
    optimizer,loss,TEM_trainable_variables,out_Y=TEM_Train(X_feature,Y_feature,LR,istrain,config)

    """ Init tf""" 
    model_saver=tf.train.Saver(var_list=TEM_trainable_variables,max_to_keep=80)
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    tf_config.log_device_placement =True
    sess=tf.InteractiveSession(config=tf_config)
    tf.global_variables_initializer().run()

    train_info={"loss":[]}
    val_info={"loss":[]}
    info_keys=train_info.keys()
    best_val_cost = 1000000

    nowTime=datetime.datetime.now().strftime('%Y-%m-%d-%H-%M')
    save_file = 'log-'+nowTime+'.txt'
    f = open(save_file, 'w+')
    f.write('learning_rates: [0.003]*2000 \n')
    f.write('epoch: ' + str(config.training_epochs) + '\n')
    f.write('batch_size: '+ str(config.batch_size) + '\n')
    f.write('==============loss================\n')
    for epoch in range(0,config.training_epochs):
        """ Training""" 
        sess.run(init_train_op)
        mini_info={"loss":[]}
        i = 1
        while True:
            try:
                _,out_loss=sess.run([optimizer,loss], feed_dict={LR:config.learning_rates[epoch], istrain:True})
            except tf.errors.OutOfRangeError:
                print('end!')
                break
            except tf.errors.DataLossError:
                print('end!')
                break
            else:
                for key in info_keys:
                    mini_info[key].append(out_loss[key])
            i = i+1
        for key in info_keys:
            train_info[key].append(np.mean(mini_info[key]))

        """ Validation""" 
        sess.run(init_val_op)
        mini_info={"loss":[]}
        i = 1
        while True:
            try:
                out_loss=sess.run(loss, feed_dict={LR:config.learning_rates[epoch], istrain:False})  
            except tf.errors.OutOfRangeError:
                print('end!')
                break
            except tf.errors.DataLossError:
                print('end!')
                break
            else:
                for key in info_keys:
                    mini_info[key].append(out_loss[key])
            i = i+1
        for key in info_keys:
            val_info[key].append(np.mean(mini_info[key]))

        nowTime=datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        train_info_str = "Epoch-%d Train Loss - %.09f" %(epoch,train_info["loss"][-1])
        val_info_str = "Epoch-%d Val   Loss - %.09f" %(epoch,val_info["loss"][-1])
        print(nowTime)
        f.write(nowTime+'\n')
        print(train_info_str)
        f.write(train_info_str+'\n')
        print(val_info_str)
        f.write(val_info_str+'\n')
        
        """ save model """ 
        if epoch%20==0:
            model_name = "models/TEM/tem_model_"+str(epoch)
            model_saver.save(sess,model_name)
        model_saver.save(sess,"models/TEM/tem_model_checkpoint")
        if val_info["loss"][-1]<best_val_cost:
            best_val_cost = val_info["loss"][-1]
            model_saver.save(sess,"models/TEM/tem_model_best")
        