# 仓库说明

## 仓库说明 

这是一个yolov5_obb模型的C++版本的推理程序，推理框架选用的是onnxruntime。  
(This is a YOLOv5_obb model's C++ version's inference program uses onnxruntime as the inference framework.)

本仓库的模型，来自于这个仓库。  
[1] https://github.com/hukaixuan19970627/yolov5_obb

模型训练环境搭建，与模型训练，部署参考下面的帖子：  
[2] https://github.com/hukaixuan19970627/yolov5_obb/issues/615 (Support yolov5_obb deployment in python/c++【从数据制作到终端部署保姆级教学】 · Issue #615 · hukaixuan19970627/yolov5_obb 
 
[3] https://blog.csdn.net/qq_44783689/article/details/129843481 (yolov5_obb报错合集-CSDN博客)  


本仓库的代码，大部分来自于下面的两个仓库，基于下面的仓库修改的，感谢两位作者的开源。  

[4] https://github.com/CVHub520/yolov5_obb/tree/main/deploy  

[5] https://github.com/UNeedCryDear/yolov8-opencv-onnxruntime-cpp  


## 速度说明

模型：yolov5n_obb，imgsz=800，  
输入图片=800x500x3  
硬件：CPU: Intel(R) Core(TM) i7-9700 CPU @ 3.00GHz   3.00 GHz 16G。 GPU: GTX 1650 4G。  
本仓库代码，使用GPU推理，完整推理时间（包含前处理、推理、后处理），共计33ms左右。使用CPU推理，完整推理时间在130ms左右。  
上面链接4的代码，使用opencv推理，在CPU上运行，完整推理时间在370ms左右。  
