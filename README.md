# kaggle-project

## 项目主题--[tensorflow-speech-recognition-challenge](https://www.kaggle.com/c/tensorflow-speech-recognition-challenge)


本主题是语音识别.要求训练好后可以在树莓派3上运行,因此对计算复杂度还是有一定要求.

## 分支策略

由于没几个人咱就用主干开发分支发布的代码分支策略得了

+ master 最近的可用版本分支
+ dev 开发主干
+ test 测试代码也就是树莓派上跑的代码
+ 功能名/核心名分支 各自研究的算法等的分支,看好了需要的话merge到dev分支
+ 0.xxx等版本号

## 迭代模式

项目每周迭代,固定在周日晚上7点使用skype开个迭代会议,用于:

+ 回顾本次迭代,各自讲下自己的进展
+ 讨论所得
+ 计划下个迭代,分配任务

## 代码规范

+ 使用python3.5/3.6
+ 使用google风格的注释方式
+ 使用autopep8做格式化
+ 原型开发还是用keras来吧...

## 数据存储位置

数据集可以去kaggle项目上下载.解压后都统一存在项目目录的`dataset/`文件夹下,注意看下`.gitignore`有没有把这个文件夹放里面,防止push的时候出问题.

+ `dataset/train/`训练集
+ `dataset/test/`测试集
+ `dataset/sample_submission.csv`标注

## 文章收集

<https://github.com/NLP-Deeplearning-Club/kaggle-project/wiki/papers>



