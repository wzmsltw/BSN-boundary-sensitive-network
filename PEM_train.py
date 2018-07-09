# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import PEM_load_data

def abs_smooth(x):
    """Smoothed absolute function. Useful to compute an L1 smooth error.
    """
    absx = tf.abs(x)
    minx = tf.minimum(absx, 1)
    r = 0.5 * ((absx - 1) * minx + absx)
    return r

def PEM_loss(anchors_iou,match_iou,config):

    # iou regressor
    u_hmask=tf.cast(match_iou>0.6,dtype=tf.float32)
    u_mmask=tf.cast(tf.logical_and(match_iou<=0.6,match_iou>0.2),dtype=tf.float32)
    u_lmask=tf.cast(match_iou<0.2,dtype=tf.float32)
    
    num_h=tf.reduce_sum(u_hmask)
    num_m=tf.reduce_sum(u_mmask)
    num_l=tf.reduce_sum(u_lmask)

    r_m= config.u_ratio_m * num_h/(num_m)
    r_m=tf.minimum(r_m,1)
    u_smmask=tf.random_uniform([tf.shape(u_hmask)[0]],dtype=tf.float32)
    u_smmask=u_smmask*u_mmask
    u_smmask=tf.cast(u_smmask > (1. - r_m), dtype=tf.float32)    
    
    r_l= config.u_ratio_l * num_h/(num_l)
    r_l=tf.minimum(r_l,1)
    u_slmask=tf.random_uniform([tf.shape(u_hmask)[0]],dtype=tf.float32)
    u_slmask=u_slmask*u_lmask
    u_slmask=tf.cast(u_slmask > (1. - r_l), dtype=tf.float32)  
    
    iou_weights=u_hmask+u_smmask+u_slmask
    iou_loss=abs_smooth(match_iou-anchors_iou)
    print match_iou.get_shape(),anchors_iou.get_shape()
    iou_loss=tf.losses.compute_weighted_loss(iou_loss,iou_weights)
    
    num_iou=[tf.reduce_sum(u_hmask),tf.reduce_sum(u_smmask),tf.reduce_sum(u_slmask)]
    loss={'iou_loss':iou_loss,'num_iou':num_iou}    
    return loss

def PEM_Train(X,Y_iou,LR,config):

    net=0.1*tf.matmul(X, config.W["iou_0"]) + config.biases["iou_0"]
    net=tf.nn.relu(net)
    net=0.1*tf.matmul(net, config.W["iou_1"]) + config.biases["iou_1"]
    net=tf.nn.sigmoid(net)
    
    anchors_iou=net
    anchors_iou=tf.reshape(anchors_iou,[-1])
    loss=PEM_loss(anchors_iou,Y_iou,config)
    PEM_trainable_variables=tf.trainable_variables()

    l2 = 0.0001 * sum(tf.nn.l2_loss(tf_var) for tf_var in PEM_trainable_variables)
    cost=10*loss["iou_loss"]+l2
    optimizer=tf.train.AdamOptimizer(learning_rate=LR).minimize(cost,var_list=PEM_trainable_variables)
    loss["l2"]=l2
    return optimizer,loss
    
class Config(object):
    """
    define a class to store parameters,
    the input should be feature mat of training and testing
    """
    def __init__(self):
        #common information
        self.training_epochs = 61
        
        self.input_steps=256
        self.learning_rates=[0.001]*10+[0.0001]*10
        self.training_epochs = len(self.learning_rates)
        #self.lambda_loss_amount = 0.000075
        
        self.num_random=10
        self.batch_size=16
        self.u_ratio_m=1
        self.u_ratio_l=2
        
        with tf.variable_scope("latent_net"):
            self.W = {
                'iou_0': tf.Variable(tf.truncated_normal([32, 256])),
                'iou_1': tf.Variable(tf.truncated_normal([256, 1]))}
            self.biases = {
                'iou_0': tf.Variable(tf.truncated_normal([256])),
                'iou_1': tf.Variable(tf.truncated_normal([1]))}


if __name__ == "__main__":
    config = Config()
    
    X_feature= tf.placeholder(tf.float32, [None,32])
    Y_iou=tf.placeholder(tf.float32,[None])
    LR= tf.placeholder(tf.float32)
    
    latent_optimizer,loss=PEM_Train(X_feature,Y_iou,LR,config)
    
    model_saver=tf.train.Saver(var_list=tf.trainable_variables(),max_to_keep=80)
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    tf_config.log_device_placement =True
    sess=tf.InteractiveSession(config=tf_config)
    tf.global_variables_initializer().run()  

    train_dict,val_dict,test_dict=PEM_load_data.getDatasetDict()

    train_data=PEM_load_data.getTrainData(config.batch_size,"train")
    val_data=PEM_load_data.getTrainData(config.batch_size,"validation")

    train_info={"iou_loss":[],"l2":[]}
    val_info={"iou_loss":[],"l2":[]}

    best_val_cost = 1000000
    
    for epoch in range(0,config.training_epochs):
    ## TRAIN ##
        
        mini_info={"iou_loss":[],"l2":[]}
        for idx in range(len(train_data)):
            prop_dict=train_data[idx]
            batch_feature,batch_iou_list,batch_ioa_list=PEM_load_data.prop_dict_data(prop_dict)
            _,out_loss,out_alpha=sess.run([latent_optimizer,loss,config.alpha],feed_dict={X_feature:batch_feature,
                                                              Y_iou:batch_iou_list,
                                                              LR:config.learning_rates[epoch]})  
            mini_info["iou_loss"].append(out_loss["iou_loss"])
            mini_info["l2"].append(out_loss["l2"])

        train_info["iou_loss"].append(np.mean(mini_info["iou_loss"]))
        train_info["l2"].append(np.mean(mini_info["l2"]))

        mini_info={"iou_loss":[],"l2":[]}
        for idx in range(len(val_data)):
            prop_dict=val_data[idx]
            batch_feature,batch_iou_list,batch_ioa_list=PEM_load_data.prop_dict_data(prop_dict)
            out_loss=sess.run(loss,feed_dict={X_feature:batch_feature,
                                                              Y_iou:batch_iou_list,
                                                              LR:config.learning_rates[epoch]})  
            mini_info["iou_loss"].append(out_loss["iou_loss"])
            mini_info["l2"].append(out_loss["l2"])

        val_info["iou_loss"].append(np.mean(mini_info["iou_loss"]))
        val_info["l2"].append(np.mean(mini_info["l2"]))
        
        print "Epoch-%d Train Loss:  %.04f" %(epoch,train_info["iou_loss"][-1])
        print "Epoch-%d Val   Loss:  %.04f" %(epoch,val_info["iou_loss"][-1])
        
        model_saver.save(sess,"models/PEM/pem_model_checkpoint")
        if val_info["iou_loss"][-1]<best_val_cost:
            best_val_cost = val_info["iou_loss"][-1]
            model_saver.save(sess,"models/PEM/pem_model_best")