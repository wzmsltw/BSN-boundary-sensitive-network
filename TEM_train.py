# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import TEM_load_data

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
    
    loss = coef_1*pmask*tf.log(pred_anchors)+coef_0*(1.0-pmask)*tf.log(1.0-pred_anchors)
    loss = -tf.reduce_mean(loss)
    num_sample = [tf.reduce_sum(pmask),ratio] 
    return loss,num_sample

def TEM_loss(anchors_action,anchors_start,anchors_end,
                  Y_action,Y_start,Y_end,config):
    """Calculateloss for action, start and end saparetely
    """  
    loss_action,num_sample_action = binary_logistic_loss(Y_action,anchors_action)
    loss_start,num_sample_start = binary_logistic_loss(Y_start,anchors_start)
    loss_end,num_sample_end = binary_logistic_loss(Y_end,anchors_end)

    loss={"loss_action":loss_action,"num_sample_action":num_sample_action,
          "loss_start":loss_start,"num_sample_start":num_sample_start,
          "loss_end":loss_end,"num_sample_end":num_sample_end}
    return loss
    

def TEM_Train(X_feature,Y_action,Y_start,Y_end,LR,config):
    """ Model and loss function of temporal evaluation module
    """ 
    net=tf.layers.conv1d(inputs=X_feature,filters=512,kernel_size=3,strides=1,padding='same',activation=tf.nn.relu)
    net=tf.layers.conv1d(inputs=net,filters=512,kernel_size=3,strides=1,padding='same',activation=tf.nn.relu)
    net=0.1*tf.layers.conv1d(inputs=net,filters=3,kernel_size=1,strides=1,padding='same')
    net=tf.nn.sigmoid(net)

    anchors_action = net[:,:,0]
    anchors_start = net[:,:,1]
    anchors_end = net[:,:,2]
    
    loss=TEM_loss(anchors_action,anchors_start,anchors_end,Y_action,Y_start,Y_end,config)

    TEM_trainable_variables=tf.trainable_variables()
    l2 = 0.001 * sum(tf.nn.l2_loss(tf_var) for tf_var in TEM_trainable_variables)
    cost = 2*loss["loss_action"]+loss["loss_start"]+loss["loss_end"]+l2
    loss['l2'] = l2
    loss['cost'] = cost
    optimizer=tf.train.AdamOptimizer(learning_rate=LR).minimize(cost,var_list=TEM_trainable_variables)
    
    return optimizer,loss,TEM_trainable_variables
    
class Config(object):
    def __init__(self):
        self.input_steps=256
        self.learning_rates=[0.001]*10+[0.0001]*10
        self.training_epochs = len(self.learning_rates)
        self.n_inputs = 400
        self.batch_size = 16
        self.input_steps=100

if __name__ == "__main__":
    """ define the input and the network""" 
    config = Config()
    X_feature = tf.placeholder(tf.float32, shape=(config.batch_size,config.input_steps,config.n_inputs))
    Y_action = tf.placeholder(tf.float32, shape=(config.batch_size,config.input_steps))
    Y_start = tf.placeholder(tf.float32, shape=(config.batch_size,config.input_steps))
    Y_end = tf.placeholder(tf.float32, shape=(config.batch_size,config.input_steps))
    LR= tf.placeholder(tf.float32)
    optimizer,loss,TEM_trainable_variables=TEM_Train(X_feature,Y_action,Y_start,Y_end,LR,config)

    """ Init tf""" 
    model_saver=tf.train.Saver(var_list=TEM_trainable_variables,max_to_keep=80)
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    tf_config.log_device_placement =True
    sess=tf.InteractiveSession(config=tf_config)
    tf.global_variables_initializer().run()  

    train_dict,val_dict,test_dict=TEM_load_data.getDatasetDict()
    train_data_dict=TEM_load_data.getFullData("train")
    val_data_dict = TEM_load_data.getFullData("val")

    train_info={"cost":[],"loss_action":[],"loss_start":[],"loss_end":[],"l2":[]}
    val_info={"cost":[],"loss_action":[],"loss_start":[],"loss_end":[],"l2":[]}
    info_keys=train_info.keys()
    best_val_cost = 1000000
    
    for epoch in range(0,config.training_epochs):
        """ Training""" 
        batch_video_list=TEM_load_data.getBatchList(len(train_dict),config.batch_size)#[:10]
        
        mini_info={"cost":[],"loss_action":[],"loss_start":[],"loss_end":[],"l2":[]}
        for idx in range(len(batch_video_list)):
            batch_label_action,batch_label_start,batch_label_end,batch_anchor_feature=TEM_load_data.getBatchData(batch_video_list[idx],train_data_dict)
            _,out_loss=sess.run([optimizer,loss], feed_dict={X_feature:batch_anchor_feature,
                                                              Y_action:batch_label_action,
                                                              Y_start:batch_label_start,
                                                              Y_end:batch_label_end,
                                                              LR:config.learning_rates[epoch]})  
            for key in info_keys:
                mini_info[key].append(out_loss[key])
        for key in info_keys:
            train_info[key].append(np.mean(mini_info[key]))

        """ Validation""" 
        batch_video_list=TEM_load_data.getBatchList(len(val_dict),config.batch_size)
        mini_info={"cost":[],"loss_action":[],"loss_start":[],"loss_end":[],"l2":[]}
        for idx in range(len(batch_video_list)):
            batch_label_action,batch_label_start,batch_label_end,batch_anchor_feature=TEM_load_data.getBatchData(batch_video_list[idx],val_data_dict)
            out_loss=sess.run(loss,feed_dict={X_feature:batch_anchor_feature,
                                                              Y_action:batch_label_action,
                                                              Y_start:batch_label_start,
                                                              Y_end:batch_label_end,
                                                              LR:config.learning_rates[epoch]})  
            for key in info_keys:
                mini_info[key].append(out_loss[key])
        for key in info_keys:
            val_info[key].append(np.mean(mini_info[key]))

        print "Epoch-%d Train Loss: Action - %.02f, Start - %.02f, End - %.02f" %(epoch,train_info["loss_action"][-1],train_info["loss_start"][-1],train_info["loss_end"][-1])
        print "Epoch-%d Val   Loss: Action - %.02f, Start - %.02f, End - %.02f" %(epoch,val_info["loss_action"][-1],val_info["loss_start"][-1],val_info["loss_end"][-1])
        
        """ save model """ 
        model_saver.save(sess,"models/TEM/tem_model_checkpoint")
        if val_info["cost"][-1]<best_val_cost:
            best_val_cost = val_info["cost"][-1]
            model_saver.save(sess,"models/TEM/tem_model_best")
        