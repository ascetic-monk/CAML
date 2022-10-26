# Context-Aware Mutual Learning for Semi-Supervised Human Activity Recognition Using Wearable Sensors



![image-20221026203101604](C:\Users\84397\AppData\Roaming\Typora\typora-user-images\image-20221026203101604.png)



## Introduction 

With the increasing popularity of wearable sensors, deep-learning-based human activity recognition  (HAR) attracts great interest from  both academic and industrial fields in recent years. Nevertheless, the performance of deep HAR methods highly depends on the quality and quantity of annotations that are not so prone to obtain in HAR. This practical concern raises broad research of semi-supervised HAR. Despite the brilliant achievements, there remain three important issues to be settled: aggravation of overfitting, deviation of distribution and ignorance of contextual information. This paper proposes a novel context-aware mutual learning method for semi-supervised human activity recognition. Firstly, a semi-supervised mutual learning framework is  introduced to alleviate the overfitting of  single network. In this framework, the main and auxiliary networks are collaboratively trained with supervised information from each other. Secondly, the distribution-preserving loss, which minimizes the distance between the class distribution of predictions and that of labeled data, is proposed to hinder the deviation of the distribution. Finally, the contextual information from the neighbor sequences is  adopted through a context-aware aggregation module. This module extracts richer information from a broader range of sequences. Our method is validated on four characteristic published human activity recognition datasets: UCI, WISDM, PAMAP2 and mHealth. The experimental result shows that the proposed method achieves superior performance compared with four typical methods in semi-supervised human activity recognition.



## Installation

To start up, you need to install some packages. Our implementation is based on [PyTorch](https://pytorch.org). We recommend using `conda` to create the environment and install dependencies and all the requirements are listed in `./requirements.txt`.

## Datasets

Four datasets: UCIHAR, WISDM, PAMAP2 and mHealth are utilized in this repo. As for the preprocessing of datasets, you can feel free to :

(1) use the raw dataset from the official website and put it into the folder`./dataset_raw`. Then you are able to further preprocess it through the scripts in `./data_preprocess` as follows:

```
python wisdm_preprocess.py
```

(2) directly utilize the processed datasets in the baidu disk: https://pan.baidu.com/s/1cMuR_rq4o6cqOoP_6f-uAw, pwd: ncp0. Then unzip the zip file into the path: `./dataset`.

## Training

After the environment and the datasets are configured. You can run the code as follows:

```shell
python train.py --DATASET uci --exp caml --ratio 0.005
```

## License

We use the MIT License. Details see the LICENSE file.

## Contact Us

If you have any questions, you can turn to the issue block or send emails to 1120220290@mail.nankai.edu.cn. Glad to have a discussion with you!