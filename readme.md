# HandPoseDetection

​	该项目将目标检测跟人体姿态估计结合，以解决手部目标检测时无法完成的关于左右手区分的问题。

​	**人体姿态估计**借助[MediaPipe](https://mediapipe.dev/index.html)完成。

​	**目标检测**借助[yolov5](https://github.com/ultralytics/yolov5)完成。

## 算法模型

​	**人体姿态估计**的算法和模型由``Google``提供，未进行算法的创新以及`custom dataset`的训练。

​	**yolov5**的算法和模型选择了`yolov5s.pt`,yolov5官方给其的描述是`the smallest and the fastest model`。如果需要重新训练，可参照[table](https://github.com/ultralytics/yolov5#pretrained-checkpoints)选择网络进行训练。

​	训练之后的准确率如下：

<img src="https://gitee.com/shen_wenxin0510/readme-pictures/raw/master/1.png" alt="image-20210729221254181" style="zoom:50%;" />

### yolov5训练

​	yolov5的训练方式详见[train custom dataset](https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data).

​	附上github上另一位大佬提供的`coco`格式数据集转`yolo`格式数据集[代码](https://github.com/Weifeng-Chen/DL_tools/blob/main/coco2yolo.py).

## 项目结构

该项目的文件结构如下：

```
|-PoseHandDetection		# 项目主要文件夹
	|-demo.py			# 使用demo
|-objectDetection		# yolov5 检测使用模块
	|-models
	|-utils
	|-HandModule.py		#yolov5 detection class
	|-requiremets.txt	#yolov5环境要求
|-HumanPose
	|-PoseModule.py		humanpose detection class
|-weights				#模型
	|-hand_yolov5_best.pt
|-test_vedio			
|-result
```

## 使用方法

### 环境配置

​	在项目开始前请完成环境的配置。

·首先请参照`objectDetection/requirement.txt`，配置``yolov5``所需环境。

·然后配置`mediapipe`所需环境:`pip install mediapipe`

### 快速开始

​	在`PoseHandDetection`中提供了使用demo。

#### step 1

​	`cd PoseHandDetection`进入`PoseHandDetection`文件夹中。

#### step 2

​	``demo.py``结构

```python
	stype = "webcam"		#vedio or webcam
    if stype == "vedio":
        ....some codes
    elif stype == "webcam":
        ....some codes
```

在第``114``行附近修改`stype(webcam/vedio)`

·**若预测的是`vedio`**

​	请按实际情况修改以下`value`值

```python
	if stype == "vedio":
        """
        value:
            source : the vedio source
            draw: show the infer result
            device :  cuda device, i.e. 0 or 0,1,2,3 or cpu
            half: use FP16 half-precision inference(yolo v5)
            weights: yolov5 model
            outPutDirName : result output
        """
        draw = True
        source = "../test_vedio/test.mp4"
        device = "cpu"
        half = False
        weight = "../weights/hand_yolov5_best.pt"
        outPutDirName = '../result/pic/'
        ...some codes
```

·**若预测的是`webcam`**	

​	请按实际情况修改以下`value`值		

```python
    elif stype == "webcam":
        """
        value:
            draw: show the infer result
            device :  cuda device, i.e. 0 or 0,1,2,3 or cpu
            half: use FP16 half-precision inference(yolo v5)
            weights: yolov5 model
            outPutDirName : result output
       
        """
        draw = True
        device = "cpu"
        half = False
        weight = "../weights/hand_yolov5_best.pt"
        outPutDirName = '../result/pic/'
```

#### step 3 

`python demo.py`

### 详细介绍

​	接下来详细介绍每个类的使用方法。该项目一共封装了两个类。`PoseModule`跟`HandModule`。分别在`HumanPose`跟`objectDetection`文件夹中。

#### PoseModule

​	该`module`主要负责人体姿态估计的检测。以读取视频检测为例：

```python
from HumanPose import PoseModule as pm

pose_detector = pm.poseDetector()
cap = cv2.VideoCapture(source)
while True:
    success, img0 = cap.read()
    if not success:
        break
    #得到初步结果：
    #注意 img0为BGR图像，findPose中会自动转为bgr
    img2 = pose_detector.findPose(img0)	
    #将结果整理成lmList的格式:
    #lmList:[id,cx,cy]的List
    lmList = pose_detector.findPosition(img0)
    # 根据lmList 返回左手手腕坐标
    lhx, lhy = get_left_hand_pos(lmList) 
    # 根据lmList 返回右手手腕坐标
    rhx, rhy = get_right_hand_pos(lmList)
```

·**Note1:**lmList中，不止包含左手手腕坐标，其包括的内容以及含义如下图所示：

<img src="https://gitee.com/shen_wenxin0510/readme-pictures/raw/master/2.png" style="zoom:50%;" />

​	如果您需要其他位置的坐标，请参照`get_left_hand_pos/get_right_hand_pos`函数实现。

·**Note2:**`get_left_hand_pose`如下,列出了几个关于左手检测关键点的id，目前是直接返回左手手腕的位置，可能存在不确定性跟不准确性，如果您需要添加**计算方式**到以下几个关键点中，请在注释中添加。

```python
def get_left_hand_pos(lmList):
    """
    id      lable_name
    15      left_wrist
    17      left_pinky
    19      left_index
    21      left_thumb

    """
    left_wrist = 15

    # now I set :
    # cx = left_wrist.x
    # cy = left_wrist.y
    # if you want to add some compute method,please add here

    cx = lmList[left_wrist][1] if lmList[left_wrist][1] else 0
    cy = lmList[left_wrist][2] if lmList[left_wrist][2] else 0

    return cx, cy
```

#### HandModule

​	该`module`主要负责手势识别的检测与分类。以读取视频检测为例：

```python
from objectDetection import HandModule as hm
"""
  value:
      device :  cuda device, i.e. 0 or 0,1,2,3 or cpu
      half: use FP16 half-precision inference(yolo v5)
      weights: yolov5 model path
 """
hand_detector = hm.HandDetector(weights=weight, device=device, half=half)
hand_detector.load_model()
# imgsz inference size (pixels)
imgsz = hand_detector.check_imapredge_size(imgsz=640)
cap = cv2.VideoCapture(source)
while True:
    success, img0 = cap.read()
    if not success:
        break
    #yolov5
    # Load_imgs
    img1= hm.Load_images(img_size=imgsz, 
                         img0 = img0,
                         device=hand_detector.device)
    # 初步得到预测结果
    #classes: fillter by class
    #max_det: maximum detections per image
    pred = hand_detector.inference(classes=None, 
                                   max_det=100, 
                                   img=img1)
    #整理预测结果
    #object_result: List([lable,conf,cx,cy])
     object_result = process_prediction(
         				pred=pred, 
                        img = img1, 
                        im0=img0,
                        names = hand_detector.names)
```

·**Note1**：``process_prediction``只完整返回`pred`中的所有内容到`object_result`中，极端情况下可能存在出现一张图片将一只手检测出两种姿势的情况，可能需要按照``conf``进行提前筛选。但`process_predicton`中尚未进行以上操作，如需要请自行更改。

## 文档

### HandModule.py

​	该文件位于``objectDetection``文件夹下。

##### Load_images

​	数据集加载。

·传入参数：

1. ​	**img0**:图片
2. ​	**device**:cuda 0 1 or cpu
3. ​	**img_size**：img size
4. ​	**stride**: 就是stride
5. ​	**half**: use FP16 half-precision inference(yolo v5)

·传出参数：

​	**img**:符合格式的img

#### HandDetector

​	手势检测与分类。

​	该类生命的时候请提供以下参数

1. **weights**:model.pt path(s)
2. **device**:cuda device, i.e. 0 or 0,1,2,3 or cpu
3. **half**： use FP16 half-precision inference

##### load_model

​	请在声明了该类之后调用此函数，来加载模型

·传入参数：无

·返回值：无

##### check_imapredge_size

​	检查您想要inference的图片的size 是否符合标准

·传入参数：**imgsz**(int)

·传出参数：**new_size**(int)

##### inference

​	预测

·传入参数：

1. **classes**：fillter by class
2. **max_det**: maximum detections per image
3. **img**: 待预测图片

·传出参数：

​	**pred**:预测结果

### PoseModule.py

​	该文件位于``HumanPose``文件夹下。

#### poseDetector

​	该类声明的时候可选择提供以下参数：

**mode:**

If set to `false`, the solution treats the input images as a video stream. It will try to detect the most prominent person in the very first images, and upon a successful detection further localizes the pose landmarks. In subsequent images, it then simply tracks those landmarks without invoking another detection until it loses track, on reducing computation and latency. If set to `true`, person detection runs every input image, ideal for processing a batch of static, possibly unrelated, images. Default to `false`.

**smooth**:

If set to `true`, the solution filters pose landmarks across different input images to reduce jitter, but ignored if **mode**is also set to `true`. Default to `true`.

**detectionCon**:

Minimum confidence value (`[0.0, 1.0]`) from the person-detection model for the detection to be considered successful. Default to `0.5`.

**trackCon**:

Minimum confidence value (`[0.0, 1.0]`) from the landmark-tracking model for the pose landmarks to be considered tracked successfully, or otherwise person detection will be invoked automatically on the next input image. Setting it to a higher value can increase robustness of the solution, at the expense of a higher latency. Ignored if **mode**is `true`, where person detection simply runs on every image. Default to `0.5`.

##### findPose

​	初步预测结果

·传入参数：img(BGR格式)

·传出参数：无，结果存在``self.result``中

##### findPosition

​	根据预测的结果，整理成合适的格式

·输入参数：img

·返回参数：lmList([id,cx,cy])

## 最终结果

​	如图，可以解决左右手不分的目标识别问题。

<img src="https://gitee.com/shen_wenxin0510/readme-pictures/raw/master/44444.png" style="zoom: 25%;" />

## 参考

​	该项目参考了`google`的[MediaPipe](https://mediapipe.dev/index.html)

​	也参考了[yolov5](https://github.com/ultralytics/yolov5)

