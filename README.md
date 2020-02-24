# facenetInNode
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