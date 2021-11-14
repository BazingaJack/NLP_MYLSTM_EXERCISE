## NLP_MYLSTM_EXERCISE

### 课程项目概述
本次课程项目旨在不使用pytorch封装好的LSTM框架模块的前提下手动搭建一个LSTM模型，其中，数据加载、模型训练等代码已经事先给出，只需要实现核心的模型搭建部分即可。基于此，我利用nn..Linear、nn.Parameter以及一些python的基本数据结构完成了单层、双层和双向的LSTM模型的搭建，其核心代码逻辑主要参考pytorch官方文档，在此基础上做了一些自己的改进。就模型训练结果而言，自己手动实现的LSTM模型略逊于官方的LSTM模型，且单层、双层和双向LSTM模型之间的训练效果呈下降趋势，说明我们的模型还有待改进和优化。
### 程序说明
Simple_LSTM.py是单层LSTM模型，Two_Layer_LSTM.py是单层LSTM模型，Two_Way_LSTM.py是双向LSTM模型。
### 实验环境
python : 3.8.2 

pytorch : 1.9.0 + cpu

Visual Studio Code : 1.62.2

CPU : AMD Ryzen 7 4800U with Radeon Graphics 1.80 GHz

GPU : AMD Radeon(TM) Graphics


