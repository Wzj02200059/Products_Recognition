# 重复图片匹配
基于patch的图片匹配，可用于重复图检测，复杂场景下的图像拼接。

![image](https://github.com/Wzj02200059/Products_Recognition/blob/master/duplication/demo/intro.png)

### 1.功能介绍
- **Opencv方法**\
  在传统特征点提取与匹配方法上的优化，能够实现多平面2D图像的重复性检测。
- **Deep Match**\
  基于深度神经网络的匹配方法，通过检测网络提取感兴趣目标的位置与特征，再由GAT进行全局信息的消息聚合，最后通过Sinkhorn算法求解软分配矩阵，完成匹配。

### 2. 使用
使用脚本duplication.py和deep_duplication.py进行计算，得到重复的图片、相似置信度。使用方式如下：
 ```python
python duplication.py --path 图片文件
    --homo_diff \ # 可选项，计算两个homo之间的dist
    --iou \ # 可选项，计算mask与原图之间的IOU
    --gpu \ # 可选项，使用gpu加速
 ```
 ```python
python deep_duplication.py --path 图片文件
    --homo_diff \ # 可选项，计算两个homo之间的dist
    --iou \ # 可选项，计算mask与原图之间的IOU
    --det_model_name\ # 必选项，选择检测模型
    --det_model\ # 必选项，检测模型的权重路径
    --portrait \ # 可选项，基于高级语义信息，识别特写照(所谓不重复的重复图)。
    --gpu \ # 可选项，使用gpu加速
 ```
### 3. Motivation
在对图片进行处理与识别之前，需要先将图片去重。
当图片中包含多个平面且伴随着较大的角度变化时。基于kps&des匹配，然后根据matches、homograph、mask的匹配方法，就不Work了。Like:

![image](https://github.com/Wzj02200059/Products_Recognition/blob/master/duplication/demo/demo.jpg)  

主要是由于2D图片中存在多个平面、在计算mask的internal points和external points时，不可避免地会丢失其他平面的点。
为了解决这个问题。在计算出第一个homo之后，从match上的goods里剔掉所有的inliers，针对outliers，再算一次homo，合并两个homo的inliers，能解决绝大部分的重复图漏检情况。
 
![image](https://github.com/Wzj02200059/Products_Recognition/blob/master/duplication/demo/first_plane.jpg)   

![image](https://github.com/Wzj02200059/Products_Recognition/blob/master/duplication/demo/second_plane.jpg)   

### 4. Motivation之GAT
在密集场景下，由于高度相似同类型商品的存在，基于low-level的特征点常常会形成错误的匹配对而导致结果错误。本模块旨在解决这个问题。     

Before:
![image](https://github.com/Wzj02200059/Products_Recognition/blob/master/duplication/demo/sift_result.jpg) 
After:
![image](https://github.com/Wzj02200059/Products_Recognition/blob/master/duplication/demo/det_superglue_result.jpg)       

主要是由于不同的实例间建立了错误的匹配。  
要解决这个问题，引入全局的实例间的结构信息是必须的。   
因此，我引入了检测模型，将每一个BoundingBox的embedding与kps信息进行融合，把每个des的kpt的位置信息，encoing之后加回到des。     
同时引入GAT，来建模所有des之间的结构关系，这个embedding也加回到des，进行消息聚合。     
这样，我们就可以认为des具备了原有的局部特征外，还具备了位置信息、以及全局的结构信息。根据相似性对两组图片的des进行match后，在幸存下来的inliers中，再剔掉所有category不一致的【可选项】。   
就可以解决上图所示的问题拉。
match的详细过程如下笔记：   

![image](https://github.com/Wzj02200059/Products_Recognition/blob/master/duplication/demo/note.jpg)

### 4. TODO
- 添加训练脚本
- Det模块与GAT模块的梦幻联动

### 4. 参考文献
- 《SuperPoint: Self-Supervised Interest Point Detection and Description》
- 《Learning Combinatorial Embedding Networks for Deep Graph Matching》
- 《SuperGlue: Learning Feature Matching with Graph Neural Networks》