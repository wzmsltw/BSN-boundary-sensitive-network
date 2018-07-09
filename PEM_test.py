# -*- coding: utf-8 -*-
"""
Created on Fri Feb 17 20:25:55 2017

@author: wzmsltw
"""

import tensorflow as tf
import numpy as np
import pandas as pd
import  PEM_load_data


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
                'iou_0': tf.Variable(tf.truncated_normal([32, 256])),
                'iou_1': tf.Variable(tf.truncated_normal([256, 1]))}
            self.biases = {
                'iou_0': tf.Variable(tf.truncated_normal([256])),
                'iou_1': tf.Variable(tf.truncated_normal([1]))}


if __name__ == "__main__":
    config = Config()
    
    X_feature = tf.placeholder(tf.float32, [None,32])

    prop_score=PEM_inference(X_feature,config)
    
    model_saver=tf.train.Saver(var_list=tf.trainable_variables(),max_to_keep=80)
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    tf_config.log_device_placement =True
    sess=tf.InteractiveSession(config=tf_config)
    tf.global_variables_initializer().run()  
    model_saver.restore(sess,"models/PEM/pem_model_best")  

    train_dict,val_dict,test_dict=PEM_load_data.getDatasetDict()

    FullDict=PEM_load_data.getTestData("validation")

    batch_video_list=PEM_load_data.getBatchList(val_dict,config.batch_size)
    video_list=val_dict.keys()
    for idx in range(len(video_list)):
        video_name=video_list[idx]
        prop_dict=FullDict[video_name]
        batch_feature,batch_iou_list,batch_ioa_list=PEM_load_data.prop_dict_data({"data":prop_dict})
        out_score=sess.run(prop_score,feed_dict={X_feature:batch_feature})  
                                                          
        out_score=np.reshape(out_score,[-1])
        xmin_list=prop_dict["xmin"]
        xmax_list=prop_dict["xmax"]
        xmin_score_list=prop_dict["xmin_score"]
        xmax_score_list=prop_dict["xmax_score"]
        latentDf=pd.DataFrame()
        latentDf["xmin"]=xmin_list
        latentDf["xmax"]=xmax_list
        latentDf["xmin_score"]=xmin_score_list
        latentDf["xmax_score"]=xmax_score_list
        latentDf["iou_score"]=out_score
        
        
        latentDf.to_csv("./output/PEM_results/"+video_name+".csv",index=False)

