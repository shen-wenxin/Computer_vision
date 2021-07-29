"""
    this is a demo about how to use the class of HumanPose and objectDetection
    author: Wenxin
    latest date: 2021-7-28
"""
import cv2
import time
import sys
import torch
sys.path.append("../")
from HumanPose import PoseModule as pm
from objectDetection import HandModule as hm
from objectDetection.utils.datasets import LoadImages

from objectDetection.utils.general import xyxy2xywh,scale_coords

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

    cx = lmList[left_wrist][1]
    cy = lmList[left_wrist][2]

    return cx, cy
def get_right_hand_pos(lmList):
    """
    id      lable_name
    16      right_wrist
    18      right_pinky
    20      right_index
    22      right_thumb

    """
    right_wrist = 16

    # now I set :
    # cx = left_wrist.x
    # cy = left_wrist.y
    # if you want to add some compute method,please add here

    cx = lmList[right_wrist][1]
    cy = lmList[right_wrist][2]

    return cx, cy
def process_prediction(pred, img, im0, names):
    result = []
    for i, det in enumerate(pred):  # detections per image
        gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]

        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
            # Write results
            for *xyxy, conf, cls in reversed(det):
                xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh

                cx, cy = int(xywh[0] * 1280), int(xywh[1] * 720)
                lable = names[int(cls)]
                confi = f'{conf:.2f}'
                result.append([lable, confi, cx, cy])

    return result
def draw_result(object_result, rx,ry,lx,ly,img):
    """

    :param object_result: yolov5's detect result([lable, confi, cx, cy])
    :param rx: right hand x
    :param ry: right hand y
    :param lx: left hand x
    :param ly: left hand y
    :return:
    """

    # draw human pose left hand
    cv2.circle(img, (lx, ly), 9, (25, 25, 112), cv2.FILLED)
    cv2.putText(img, "left_wrist", (lx, ly), cv2.FONT_HERSHEY_PLAIN, 2,
                (25, 25, 112), 2)
    cv2.circle(img, (rx, ry), 9, (205, 92, 92), cv2.FILLED)
    cv2.putText(img, "right_wrist", (rx, ry), cv2.FONT_HERSHEY_PLAIN, 2,
                (205, 92, 92), 2)

    # draw yolov5 result
    for ele in object_result:

        lable = ele[0]
        conf = ele[1]
        cx = ele[2]
        cy = ele[3]
        s = "{}:{}".format(lable,conf)
        cv2.circle(img, (cx, cy), 8, (0, 100, 0), cv2.FILLED)
        cv2.putText(img, s, (cx, cy), cv2.FONT_HERSHEY_PLAIN, 2,
                    (0, 100, 0), 2)






if __name__ == '__main__':
    # source type: vedio or img or webcam.give the example of vedio and webcam
    stype = "vedio"
    # demo1:vedio detection
    if stype == "vedio":
        """
        value:
            source : the vedio source
            draw: show the infer result
            device :  cuda device, i.e. 0 or 0,1,2,3 or cpu
            half: use FP16 half-precision inference(yolo v5)
       
        """
        draw = True
        source = "../test_vedio/test.mp4"
        device = "1"
        half = False
        weight = "../weights/hand_yolov5_best.pt"

        outPutDirName = '../result/'

        fourcc = cv2.VideoWriter_fourcc(*'MP4V')
        out = cv2.VideoWriter('result.mp4', fourcc, 30, (1280, 740))  # 写入视频

        cout = 0
        pTime = 0
        pose_detector = pm.poseDetector()   #human pose detector


        hand_detector = hm.HandDetector(weights=weight, device=device, half=half)
        hand_detector.load_model()
        imgsz = hand_detector.check_imapredge_size(imgsz=640)# inference size (pixels)


        cap = cv2.VideoCapture(source)

        while True:
            success, img0 = cap.read()
            if not success:
                break

            #yolov5
            img1= hm.Load_images(img_size=imgsz, img0 = img0,device=hand_detector.device)
            pred = hand_detector.inference(classes=None, max_det=100, img=img1)
            object_result = process_prediction(pred=pred, img = img1, im0=img0,
                                               names = hand_detector.names)
            print(object_result)

            #mediapipe
            img2 = pose_detector.findPose(img0)
            lmList = pose_detector.findPosition(img0)
            lhx, lhy = get_left_hand_pos(lmList)    #left humen pose hand x,y
            rhx, rhy = get_right_hand_pos(lmList)     #left humen pose hand x,y

            # draw the result
            draw_result(object_result= object_result, rx = rhx,ry=rhy, lx=lhx, ly= lhy, img =img0)



            #time
            cTime = time.time()
            fps = 1 / (cTime - pTime)
            pTime = cTime

            cv2.putText(img0, str(int(fps)), (70, 50), cv2.FONT_HERSHEY_PLAIN, 3,
                        (255, 0, 0), 3)

            cv2.imwrite(outPutDirName + str(cout)+'.jpg', img0)
            cout = cout + 1
            cv2.imshow('Image', img0)

            #save vedio

            cv2.waitKey(1)

        cap.release()








