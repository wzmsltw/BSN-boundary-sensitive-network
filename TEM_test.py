import tensorflow as tf
import numpy as np
import pandas as pd
import TEM_load_data

def TEM_inference(X_feature,config):
    net=tf.layers.conv1d(inputs=X_feature,filters=512,kernel_size=3,strides=1,padding='same',activation=tf.nn.relu)
    net=tf.layers.conv1d(inputs=net,filters=512,kernel_size=3,strides=1,padding='same',activation=tf.nn.relu)
    net=0.1*tf.layers.conv1d(inputs=net,filters=3,kernel_size=1,strides=1,padding='same')
    net=tf.nn.sigmoid(net)

    anchors_action=net[:,:,0]
    anchors_start=net[:,:,1]
    anchors_end=net[:,:,2]
    
    scores={"anchors_action":anchors_action,
            "anchors_start":anchors_start,
            "anchors_end":anchors_end}
    return scores
    
class Config(object):
    def __init__(self):
        #common information
        self.training_epochs = 61
        self.input_steps=256
        self.learning_rates=[0.001]*10+[0.0001]*10
        self.n_inputs = 400
        self.batch_size = 16
        self.input_steps=100

if __name__ == "__main__":
    config = Config()
    X_feature = tf.placeholder(tf.float32, shape=(config.batch_size,config.input_steps,config.n_inputs))
    tem_scores=TEM_inference(X_feature,config)
    
    model_saver=tf.train.Saver(var_list=tf.trainable_variables(),max_to_keep=80)
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    tf_config.log_device_placement =True
    sess=tf.InteractiveSession(config=tf_config)
    tf.global_variables_initializer().run()  
    model_saver.restore(sess,"models/TEM/tem_model_best")  

    video_dict= TEM_load_data.load_json("./data/activitynet_annotations/anet_anno_action.json")

    batch_result_action=[]
    batch_result_start=[]
    batch_result_end=[]
    batch_result_xmin=[]
    batch_result_xmax=[]
    
    batch_video_list=TEM_load_data.getBatchListTest(video_dict,config.batch_size,shuffle=False)
    
    for idx in range(len(batch_video_list)):
        batch_anchor_xmin,batch_anchor_xmax,batch_anchor_feature=TEM_load_data.getProposalDataTest(batch_video_list[idx],video_dict)
        out_scores=sess.run(tem_scores,feed_dict={X_feature:batch_anchor_feature})  
        batch_result_action.append(out_scores["anchors_action"])
        batch_result_start.append(out_scores["anchors_start"])
        batch_result_end.append(out_scores["anchors_end"])
        batch_result_xmin.append(batch_anchor_xmin)
        batch_result_xmax.append(batch_anchor_xmax)

    columns=["action","start","end","xmin","xmax"]

    for idx in range(len(batch_video_list)):
        b_video=batch_video_list[idx]
        b_action=batch_result_action[idx]
        b_start=batch_result_start[idx]
        b_end=batch_result_end[idx]
        b_xmin=batch_result_xmin[idx]
        b_xmax=batch_result_xmax[idx]
        for j in range(len(b_video)):
            tmp_video=b_video[j]
            tmp_action=b_action[j]
            tmp_start=b_start[j]
            tmp_end=b_end[j]
            tmp_xmin=b_xmin[j]
            tmp_xmax=b_xmax[j]
            tmp_result=np.stack((tmp_action,tmp_start,tmp_end,tmp_xmin,tmp_xmax),axis=1)
            tmp_df=pd.DataFrame(tmp_result,columns=columns)  
            tmp_df.to_csv("./output/TEM_results/"+tmp_video+".csv",index=False)