﻿# facenetInNode
Most of code in ./public/javascripts\<br>
* 2020.01.16 finished alignFace.js and tested the result.\<br>
* 2020.02.10 add SVMClassifier.js to test SVMClassifier and use process to show load process\<br>
* 2020.02.11 SVMClassifier.js fineturn the factor of SVM.change different style of process bar\<br>
* 2020.02.12 微调SVM参数，尝试SVM的多分类有向图构建\<br>
* 2020.02.13 toolMatrix.js 改变算法，修改了因图片过小而导致的问题 facenet.js 提供了判定的方法，将带有透明度的图片张量重新组织数据。 
* 2020.02.14 SVMClassifier.js 采用新的方法训练和组织SVM分类器
* 2020.02.17 SVMClassifier.js 测试分类器
* 2020.02.18 SVMClassifier.js 尝试使用递归函数构造分类数
* 2020.02.19 SVMClassifier.js 导出json格式的树形分类器，并且可以二次加载。
* 2020.02.20 SVMClassifier.js 了解了GAN的训练方式。利用树形分类器得出预测结果。
* 2020.02.21 SVMClassifier.js 编写函数来调用该结构完成对结果的分类
* 2020.02.24 进行了不同数据集上的测试，继续了解GAN的构建思路。
* 2020.02.25 GAN.js 进行简单DCGAN的尝试。
* 2020.02.26 GAN.js DCGAN的generator model和discriminator model的构建。
* 2020.02.27 GAN.js 尝试从mnist数据集中进行小型DCGAN的训练。
* 2020.02.28 GAN.js 初步了解有关人脸修复GAN目前应用的类型。
* 2020.03.03 GAN.js 加载minist数据集，了解PI-GAN，LB-GAN，BoostGAN，关于面部转正的方法。
* 2020.03.04 DCGan.js 使用类的方法重新构建Gan函数。
* 2020.03.05 DCGan.js 继续进行train方法和save_img方法的实现。
* 2020.03.06 DCGan.js 采用randomNumArr方法随机的获取mnist训练集,对于初始噪声采用从正态分布中采样的值
* 2020.03.09 DCGan.js 修改了模型中的一些错误，使训练结果趋于相似。
* 2020.03.10 查阅了一些关于GAN实现异常检测的资料，着手利用anoGAN实现异常检测。
* 2020.03.11 anoGAN.js 阅读了anoGAN的keras实现，完成anoGAN网络的搭建。
* 2020.03.12 anoGAN.js compute_anomaly_score()异常得分函数。
* 2020.03.13 anoGAN.js 完成训练函数，在minist数据集上训练，loss在2epoch之后趋于稳定。
* 2020.04.03 agePredict.js 利用python的预训练模型完成对图片年龄，性别的估测。