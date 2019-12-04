# pytorch-image-classification
1、项目介绍：
 --------
适合小白入门的图像分类项目，从熟悉到熟练图像分类的流程，搭建自己的分类网络结构以及在 pytorch 中运用经典的分类网络。利用gui图形化界面进行测试单张图片。代码注释清楚，很容易理解。详细介绍可以访问[`我的博客`](https://blog.csdn.net/weixin_43962659/article/details/103381731)

2、环境：
-----
* pytorch 1.2.0
* python3 以上
* wxpython ：安装方式 conda install wxpython
* opencv-python  

3、数据准备：    
-----
下载数据集即四类花的分类，然后解压放到文件夹data里。  
文件夹树结构：  
* ./pytorch-image-classification
  * data
    * input_data
  * example
  * checkpoints
  * logs
  * utils
  * train.py Model.py READEME.md 等根目录的文件  

4、快速开始：  
---------
下载本项目：git clone https://github.com/lilei1128/pytorch-image-classification.git  
修改config.py中设置的路径和其他参数。  
运行train.py进行训练，也可自行修改model中的网络结构再训练。  
最后进行测试。




