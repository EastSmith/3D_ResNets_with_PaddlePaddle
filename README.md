# **3D_ResNets_with_PaddlePaddle**
# **开始之前**
* ### 本项目是用“视频分类综述－实践作业”的项目文件改的，并没有完全按原论文的代码来
* ### 深度学习的CV技术相对来说比较成熟，model的层数少则几十层，多则上百层，学CV更容易体会什么叫做深度学习神经网络，而且选的这个文章就是解决深层的网络梯度消失、梯度爆炸等问题导致的模型难以收敛、难以训练的问题，所以模型层数只多不少的。
* ### 视频分类的数据集贼大，动辄上百G，小规模的数据集也有，只是深度学习模型在上面容易过拟合而已。
* ### 论文代码中各种预处理、后处理、辅助代码，体会到了顶会级别的论文代码的复杂性。
* ### 百度顶会论文复现营——学习心得：https://zhuanlan.zhihu.com/p/208181811
* ### AiStudio 项目地址 ：https://aistudio.baidu.com/aistudio/projectdetail/707154
* ### 课程地址 ：https://aistudio.baidu.com/aistudio/course/introduce/1340
* ### 原论文代码地址：https://github.com/kenshohara/3D-ResNets-PyTorch



# **解压数据集**
### UCF-101

* Download videos and train/test splits [here](http://crcv.ucf.edu/data/UCF101.php).

```
!unzip -q  /home/aistudio/data/data48916/UCF-101.zip  -d data1
```
# **视频抽帧**
```
!python avi2jpg.py
```
# 按UCF101_split01划分数据集
```
!python jpg2pkl.py
!python data_list_gener.py
```
# **将pytorch权重文件文件转换成paddle的权重**
```
!python pytorch2paddle.py

```
# **开始训练**
```
!python train.py --use_gpu True --epoch 2 --save_dir 'checkpoints_models_s0'  --pretrain True

```
# **开始评估**
```
!python eval.py --weights 'checkpoints_models_s0/res3d_model' --use_gpu True

```
