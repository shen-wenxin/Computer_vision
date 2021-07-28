import argparse
import sys
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn

FILE = Path(__file__).absolute()
sys.path.append(FILE.parents[0].as_posix())  # add yolov5/ to path

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, colorstr, non_max_suppression, \
    apply_classifier, scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path, save_one_box
from utils.plots import colors, plot_one_box
from utils.torch_utils import select_device, load_classifier, time_sync

@torch.no_grad()
class HandDetector():
    def __init__(self,
                 weights='weights/best.pt',  # model.pt path(s)
                 source='test.mp4',  # file/dir/URL/glob, 0 for webcam
                 device='0',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
                 line_thickness=3,  # bounding box thickness (pixels)
                 hide_labels=False,  # hide labels
                 hide_conf=False,  # hide confidences
                 half=False,  # use FP16 half-precision inference
                 ):
        self.weights = weights
        self.source = source
        self.line_thickness = line_thickness
        self.hide_labels = hide_labels
        self.hide_conf = hide_conf
        self.half = half

        # 0 : webcam
        self.webcam = source.isnumeric()  or source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))

        # Initialize
        set_logging()
        self.device = select_device(device)
        self.half &= self.device.type != 'cpu'  # half precision only supported on CUDA

        # Load model
    def load_model(self):
        self.pt = self.weights.endswith('.pt')  # inference type
        self.model = attempt_load(self.weights, map_location=self.device)  # load FP32 model
        self.stride = int(self.model.stride.max())  # model stride
        self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names  # get class names
        if self.half:
            self.model.half()  # to FP16

    def check_imapredge_size(self,
                         imgsz=640,  # inference size (pixels)
                        ):
        new_size = check_img_size(imgsz, s=self.stride)  # check image size
        return new_size

    def inference(self,
                  classes,max_det,img
                  ):
        # Inference
        pred = self.model(img, augment=False, visualize=False)[0]

        # NMS
        conf_thres = 0.25
        iou_thres = 0.45
        agnostic_nms = False
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

        return pred



if __name__ == "__main__":
    device = "0"# cuda device, i.e. 0 or 0,1,2,3 or cpu
    view_img = True
    half = False  # use FP16 half-precision inference
    hand_detector = HandDetector(device=device, half= half)
    hand_detector.load_model()
    imgsz = hand_detector.check_imapredge_size(imgsz=640)
    source_path = 'test.mp4'
    webcam = source_path.isnumeric() or source_path.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))

    dataset = LoadImages(source_path, imgsz)
    for path, img, im0s, vid_cap in dataset:


        img = torch.from_numpy(img).to(hand_detector.device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if len(img.shape) == 3:
            img = img[None]  # expand for batch dim

        pred = hand_detector.inference(classes=None,max_det=100,img=img)
        names = hand_detector.names
        # Process predictions
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], f'{i}: ', im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s.copy(), getattr(dataset, 'frame', 0)
            p = Path(p)  # to Path
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh

            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh

                    if view_img:  # Add bbox to image
                        c = int(cls)  # integer class
                        label = f'{names[c]} {conf:.2f}'
                        line_thickness = 3  # bounding box thickness (pixels)
                        plot_one_box(xyxy, im0, label=label, color=colors(c, True), line_thickness=line_thickness)

                    cx, cy = int(xywh[0] * 1280), int(xywh[1] * 720)
                    cv2.circle(im0, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
                    print("lable:",names[int(cls)])
                    print("conf:", conf)
                    print("xywh", xywh)

            if view_img:
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

