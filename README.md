# BSN-boundary-sensitive-network

This repo holds the codes of paper: "BSN: Boundary Sensitive Network for Temporal Action Proposal Generation", which is accepted in ECCV 2018.

[[Arxiv Preprint]](http://arxiv.org/abs/1806.02964)

# Update

* 2018.07.09: Codes and feature of BSN
* 2018.07.02: Repository for BSN



# Contents

* [Paper Introduction](#paper-introduction)
* [Prerequisites](#prerequisites)
* [Code and Data Preparation](#code_and_data_preparation)
* [Training and Testing  of BSN](#training_and_testing_of_bsn)
* [Other Info](#other-info)

# Paper Introduction

 <img src="./paper_pic/eccv_overview.jpg" width = "700" alt="image" align=center />

Temporal action proposal generation is an important yet challenging problem, since temporal proposals with rich action content are indispensable for analysing real-world videos with long duration and high proportion irrelevant content. This problem requires methods not only generating proposals with precise temporal boundaries, but also retrieving proposals to cover truth action instances with high recall and high overlap using relatively fewer proposals. To address these difficulties, we introduce an effective proposal generation method, named Boundary-Sensitive Network (BSN), which adopts “local to global” fashion. Locally, BSN first locates temporal boundaries with high probabilities, then directly combines these boundaries as proposals. Globally, with Boundary-Sensitive Proposal feature, BSN retrieves proposals by evaluating the confidence of whether a proposal contains an action within its region. We conduct experiments on two challenging datasets: ActivityNet-1.3 and THUMOS14, where BSN outperforms other state-of-the-art temporal action proposal generation methods with high recall and high temporal precision. Finally, further experiments demonstrate that by combining existing action classifiers, our method significantly improves the state-of-the-art temporal action detection performance.


# Prerequisites

These code is  implemented in Tensorflow (>1.0). Thus please install tensorflow first.

To  accelerate the training speed, all input feature data are loaded in RAM first. Thus around 7GB RAM is required.


# Code and Data Preparation

## Get the code

Clone this repo with git, please use:

```
git clone https://github.com/wzmsltw/BSN-boundary-sensitive-network.git
```


## Download Datasets

We support experiments with publicly available dataset ActivityNet 1.3 for temporal action proposal generation now. To download this dataset, please use [official ActivityNet downloader](https://github.com/activitynet/ActivityNet/tree/master/Crawler) to download videos from the YouTube.

To extract visual feature, we adopt TSN model pretrained on the training set of ActivityNet, which is the challenge solution of CUHK&ETH&SIAT team in ActivityNet challenge 2016. Please refer this repo [TSN-yjxiong](https://github.com/yjxiong/temporal-segment-networks) to extract frames and optical flow and refer this repo [anet2016-cuhk](https://github.com/yjxiong/anet2016-cuhk) to find pretrained TSN model.

For convenience of training and testing, we rescale the feature length of all videos to same length 100, and we provide the rescaled feature at here [Google Cloud](https://drive.google.com/file/d/1ISemndlSDS2FtqQOKL0t3Cjj9yk2yznF/view?usp=sharing) or [Baidu Yun](https://pan.baidu.com/s/19GI3_-uZbd_XynUO6g-8YQ). If you download features from BaiduYun, please use `cat zip_csv_mean_100.z* > csv_mean_100.zip` before unzip. After download and unzip, please put `csv_mean_100` directory to `./data/activitynet_feature_cuhk/` . 

# Training and Testing  of BSN

#### 1. Training of temporal evaluation module

```
python TEM_train.py
```

We also provide trained TEM model in `./model/TEM` .


#### 2. Testing of temporal evaluation module

First, to create directories for outputs.

```
sh mkdir.sh
```

```
python TEM_test.py
```

#### 3. Proposals generation

```
sh run_pgm_proposal.sh
```

#### 4. BSP feature generation

```
sh run_pgm_feature.sh
```

#### 5. Training of proposal evaluation module

```
python PEM_train.py
```

We also provide trained PEM model in `./model/PEM` .

#### 6. Testing of proposal evaluation module

```
python PEM_test.py
```

#### 7. Post processing and generate final results

```
python Post_processing.py
```

#### 8. Eval the performance of proposals

```
python eval.py
```

# Other Info

## Citation


Please cite the following paper if you feel BSN useful to your research

```
@inproceedings{BSN2018arXiv,
  author    = {Tianwei Lin and
               Xu Zhao and
               Haisheng Su and
               Chongjing Wang and
               Ming Yang},
  title     = {BSN: Boundary Sensitive Network for Temporal Action Proposal Generation},
  booktitle   = {European Conference on Computer Vision},
  year      = {2018},
}
```


## Contact
For any question, please file an issue or contact
```
Tianwei Lin: wzmsltw@sjtu.edu.cn
```

