# -*- coding: utf-8 -*-
import cv2
import time
import threading
from threading import Thread

lock = threading.Lock()
import serial
import numpy as np
import datetime

global frame
global closed
frame = cv2.imread('/home/nvidia/test.jpg')
closed = frame

mypoints1 = np.array([[[315, 512], [200, 266], [388, 266], [640, 470], [640, 512]]])
mypoints2 = np.array([[[640, 470], [388, 266], [572, 266], [640, 300]]])
mypoints3 = np.array([[[388, 266], [235, 142], [324, 142], [572, 266]]])
mypoints4 = np.array([[[200, 266], [142, 142], [235, 142], [388, 266]]])
mypoints5 = np.array([[[142, 142], [120, 90], [220, 90], [324, 142]]])
mypoints6 = np.array([[[120, 90], [106, 60], [160, 60], [220, 90]]])

center1 = bytes([0xff, 0x01, 0x00, 0xe7, 0x01, 0xf4, 0x03, 0x20, 0x2d, 0xc6, 0xf3])
center2 = bytes([0xff, 0x01, 0x00, 0xe7, 0x03, 0xe8, 0x02, 0xbc, 0x33, 0x0a, 0xce])
center3 = bytes([0xff, 0x01, 0x00, 0xe7, 0x02, 0xbc, 0x01, 0xf4, 0x36, 0xb5, 0x86])
center4 = bytes([0xff, 0x01, 0x00, 0xe7, 0x00, 0xc8, 0x01, 0xf4, 0X36, 0xee, 0xc9])
center5 = bytes([0xff, 0x01, 0x00, 0xe7, 0x00, 0xc8, 0x01, 0x90, 0X3c, 0x4c, 0xc9])
center6 = bytes([0xff, 0x01, 0x00, 0xe7, 0x00, 0x64, 0x01, 0x2c, 0X40, 0x00, 0xb9])


class ViBe:
    def __init__(self, num_sam=20, min_match=2, radiu=20, rand_sam=16):
        self.defaultNbSamples = num_sam  # 每个像素的样本集数量，默认20个
        self.defaultReqMatches = min_match  # 前景像素匹配数量，如果超过此值，则认为是背景像素
        self.defaultRadius = radiu  # 匹配半径，即在该半径内则认为是匹配像素
        self.defaultSubsamplingFactor = rand_sam  # 随机数因子，如果检测为背景，每个像素有1/defaultSubsamplingFactor几率更新样本集和领域样本集
        self.background = 0
        self.foreground = 255

    def __buildNeighborArray(self, img):
        height, width = img.shape
        self.samples = np.zeros((self.defaultNbSamples, height, width), dtype=np.uint8)
        ramoff_xy = np.random.randint(-1, 2, size=(2, self.defaultNbSamples, height, width))
        xr_ = np.tile(np.arange(width), (height, 1))
        yr_ = np.tile(np.arange(height), (width, 1)).T
        xyr_ = np.zeros((2, self.defaultNbSamples, height, width))
        for i in range(self.defaultNbSamples):
            xyr_[1, i] = xr_
            xyr_[0, i] = yr_
        xyr_ = xyr_ + ramoff_xy
        xyr_[xyr_ < 0] = 0
        tpr_ = xyr_[1, :, :, -1]
        tpr_[tpr_ >= width] = width - 1
        tpb_ = xyr_[0, :, -1, :]
        tpb_[tpb_ >= height] = height - 1
        xyr_[0, :, -1, :] = tpb_
        xyr_[1, :, :, -1] = tpr_
        xyr = xyr_.astype(int)
        self.samples = img[xyr[0, :, :, :], xyr[1, :, :, :]]

    def ProcessFirstFrame(self, img):
        self.__buildNeighborArray(img)
        self.fgCount = np.zeros(img.shape)
        self.fgMask = np.zeros(img.shape)

    def Update(self, img):
        '''
        处理每帧视频，更新运动前景，并更新样本集。该函数是本类的主函数
        输入：灰度图像
        '''
        height, width = img.shape
        dist = np.abs((self.samples.astype(float) - img.astype(float)).astype(int))
        dist[dist < self.defaultRadius] = 1
        dist[dist >= self.defaultRadius] = 0
        matches = np.sum(dist, axis=0)
        matches = matches < self.defaultReqMatches
        self.fgMask[matches] = self.foreground
        self.fgMask[~matches] = self.background
        self.fgCount[matches] = self.fgCount[matches] + 1
        self.fgCount[~matches] = 0
        fakeFG = self.fgCount > 50
        matches[fakeFG] = False
        upfactor = np.random.randint(self.defaultSubsamplingFactor, size=img.shape)  # 生成每个像素的更新几率
        upfactor[matches] = 100  # 前景像素设置为100,其实可以是任何非零值，表示前景像素不需要更新样本集
        upSelfSamplesInd = np.where(upfactor == 0)  # 满足更新自己样本集像素的索引
        upSelfSamplesPosition = np.random.randint(self.defaultNbSamples,
                                                  size=upSelfSamplesInd[0].shape)  # 生成随机更新自己样本集的的索引
        samInd = (upSelfSamplesPosition, upSelfSamplesInd[0], upSelfSamplesInd[1])
        self.samples[samInd] = img[upSelfSamplesInd]  # 更新自己样本集中的一个样本为本次图像中对应像素值

        # 更新邻域样本集
        upfactor = np.random.randint(self.defaultSubsamplingFactor, size=img.shape)  # 生成每个像素的更新几率
        upfactor[matches] = 100  # 前景像素设置为100,其实可以是任何非零值，表示前景像素不需要更新样本集
        upNbSamplesInd = np.where(upfactor == 0)  # 满足更新邻域样本集背景像素的索引
        nbnums = upNbSamplesInd[0].shape[0]
        ramNbOffset = np.random.randint(-1, 2, size=(2, nbnums))  # 分别是X和Y坐标的偏移
        nbXY = np.stack(upNbSamplesInd)
        nbXY += ramNbOffset
        nbXY[nbXY < 0] = 0
        nbXY[0, nbXY[0, :] >= height] = height - 1
        nbXY[1, nbXY[1, :] >= width] = width - 1
        nbSPos = np.random.randint(self.defaultNbSamples, size=nbnums)
        nbSamInd = (nbSPos, nbXY[0], nbXY[1])
        self.samples[nbSamInd] = img[upNbSamplesInd]

    def getFGMask(self):
        '''
        返回前景mask
        '''
        return self.fgMask


def myThread1():
    global frame
    global Flag1  # Flag1=0 代表红外未检测到目标, 1代表红外未发现目标
    global closed
    global contours
    time.sleep(30)
    vibe = ViBe()
    vibe.ProcessFirstFrame(frame)
    while True:
        if frame is None:
            continue
        gray = frame
        vibe.Update(gray)
        segMat = vibe.getFGMask()
        # 　转为uint8类型
        segMat = segMat.astype(np.uint8)
        # 形态学处理模板初始化
        kernel1 = np.ones((5, 5), np.uint8)
        # 开运算
        opening = cv2.morphologyEx(segMat, cv2.MORPH_OPEN, kernel1)
        # 形态学处理模板初始化
        kernel2 = np.ones((5, 5), np.uint8)
        # 闭运算
        closed = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel2)
        # 寻找轮廓
        contours, hierarchy = cv2.findContours(closed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) > 0:
            Flag1 = 1
        for i in range(0, len(contours)):
            x, y, w, h = cv2.boundingRect(contours[i])
            if w * h > 200 and w * h < 20000:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)


def myThread2():
    portx = "/dev/ttyUSB0"
    bps = 9600
    waitTime = 1
    ser = serial.Serial(portx, bps, timeout=waitTime)
    global contours
    global Flag1
    while True:
        if 1 == Flag1:
            AimPos = np.zeros(6)
            for i in range(0, len(contours)):
                x, y, w, h = cv2.boundingRect(contours[i])
                if 1.0 == cv2.pointPolygonTest(mypoints1, (x, y + h / 2), False):
                    AimPos[0] = 1
                    print("x1", x, "y", y, "w", w, "h", h)
                if 1.0 == cv2.pointPolygonTest(mypoints2, (x, y + h / 2), False):
                    AimPos[1] = 1
                    print("x2", x, "y", y, "w", w, "h", h)
                if 1.0 == cv2.pointPolygonTest(mypoints3, (x, y + h / 2), False):
                    AimPos[2] = 1
                    print("x3", x, "y", y, "w", w, "h", h)
                if 1.0 == cv2.pointPolygonTest(mypoints4, (x, y + h / 2), False):
                    AimPos[3] = 1
                    print("x4", x, "y", y, "w", w, "h", h)
                if 1.0 == cv2.pointPolygonTest(mypoints5, (x, y + h / 2), False):
                    AimPos[4] = 1
                    print("x5", x, "y", y, "w", w, "h", h)
                if 1.0 == cv2.pointPolygonTest(mypoints6, (x, y + h / 2), False):
                    AimPos[5] = 1
                    print("x6", x, "y", y, "w", w, "h", h)
            print("AimPos: ", AimPos)
            Flag1 = 2

        if 2 == Flag1:
            if 1 == AimPos[0]:
                AimPos[0] = 0
                ser.write(center1)
                time.sleep(3)
                print("center1")
            if 1 == AimPos[1]:
                AimPos[1] = 0
                ser.write(center2)
                time.sleep(3)
                print("center2")
            if 1 == AimPos[2]:
                AimPos[2] = 0
                ser.write(center3)
                time.sleep(3)
                print("center3")
            if 1 == AimPos[3]:
                AimPos[3] = 0
                ser.write(center4)
                time.sleep(3)
                print("center4")
            if 1 == AimPos[4]:
                AimPos[4] = 0
                ser.write(center5)
                time.sleep(3)
                print("center5")
            if 1 == AimPos[5]:
                AimPos[5] = 0
                ser.write(center6)
                time.sleep(3)
                print("center6")
            Flag1 = 0


import argparse
import os
import sys
from pathlib import Path
import math
import cv2
import torch
import torch.backends.cudnn as cudnn

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.datasets import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from utils.plots import  Annotator,colors, save_one_box
from utils.torch_utils import select_device, time_sync


@torch.no_grad()
def run(weights=ROOT / 'yolov5s.pt',  # model.pt path(s)
        source=ROOT / 'data/images',  # file/dir/URL/glob, 0 for webcam
        data=ROOT / 'data/coco128.yaml',  # dataset.yaml path
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        nosave=True,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT / 'runs/detect',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        ):
    source = str(source)
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)
    if is_url and is_file:
        source = check_file(source)  # download

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data)
    stride, names, pt, jit, onnx, engine = model.stride, model.names, model.pt, model.jit, model.onnx, model.engine
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Half
    half &= (pt or jit or onnx or engine) and device.type != 'cpu'  # FP16 supported on limited backends with CUDA
    if pt or jit:
        model.model.half() if half else model.model.float()

    # Dataloader
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt)
        bs = len(dataset)  # batch_size
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)
        bs = 1  # batch_size
    vid_path, vid_writer = [None] * bs, [None] * bs

    # Run inference
    model.warmup(imgsz=(1 if pt else bs, 3, *imgsz), half=half)  # warmup
    dt, seen = [0.0, 0.0, 0.0], 0
    for path, im, im0s, vid_cap, s in dataset:
        t1 = time_sync()
        im = torch.from_numpy(im).to(device)
        im = im.half() if half else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        t2 = time_sync()
        dt[0] += t2 - t1

        # Inference
        visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
        pred = model(im, augment=augment, visualize=visualize)
        t3 = time_sync()
        dt[1] += t3 - t2

        # NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
        dt[2] += time_sync() - t3

        # Second-stage classifier (optional)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

        # Process predictions
        for i, det in enumerate(pred):  # per image
            seen += 1
            if webcam:  # batch_size >= 1
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f'{i}: '
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # im.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
            s += '%gx%g ' % im.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if save_crop else im0  # for save_crop
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or save_crop or view_img:  # Add bbox to image
                        c = int(cls)  # integer class
                        label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                        annotator.box_label(xyxy, label, color=colors(c, True))
                        #if save_crop:
                            #save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)
                        '''
                        x1 = int(xyxy[0].item())
                        y1 = int(xyxy[1].item())
                        x2 = int(xyxy[2].item())
                        y2 = int(xyxy[3].item())
                        d = 0.15
                        r = 0.075
                        f = 0.006
                        cx = 302.617
                        cy = 225.015
                        fx = 973.078
                        fy = 974.442
                        w = x2 - x1
                        u = (x1 + x2) / 2
                        v = (y1 + y2) / 2
                        X = (2 * r * f * (u - cx)) / (fx * w)*pow(10,5)
                        Y = (2 * r * f * (v - cy)) / (fy * w)*pow(10,5)
                        Z = f * 2 * r / w *pow(10,5)
                        print("坐标为：(" + str(X) + "," + str(Y) + "," + str(Z) + ")")
                        '''
            # Stream results
            im0 = annotator.result()
            if view_img:
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:  # 'video' or 'stream'
                    if vid_path[i] != save_path:  # new video
                        vid_path[i] = save_path
                        if isinstance(vid_writer[i], cv2.VideoWriter):
                            vid_writer[i].release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                        save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer[i].write(im0)

        # Print time (inference-only)
        #LOGGER.info(f'{s}Done. ({t3 - t2:.3f}s)')

    # Print results
    t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
    #LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        #LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if update:
        strip_optimizer(weights)  # update model (to fix SourceChangeWarning)


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'yolov5n_person.engine', help='model path(s)')
    parser.add_argument('--source', type=str, default=ROOT / 'data/images', help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--data', type=str, default=ROOT / 'data/person.yaml', help='(optional) dataset.yaml path')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=ROOT / 'runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(FILE.stem, opt)
    return opt


def main(opt):
    global Flag1
    global frame
    Flag1 = 0
    global closed
    time.sleep(30)
    url = 'rtsp://admin:Admin12345@192.168.0.74/Streaming/Channels/101'
    vc = cv2.VideoCapture(url)
    if vc.isOpened():
        rval, frame0 = vc.read()
    frame = cv2.cvtColor(frame0, cv2.COLOR_BGR2GRAY)
    closed = frame
    while rval:
        if vc.isOpened():
            rval, frame0 = vc.read()
        if frame0 is None:
            continue
        frame = cv2.cvtColor(frame0, cv2.COLOR_BGR2GRAY)
    time.sleep(5)
    # 创建线程1,vibe识别
    thread_01 = Thread(target=myThread1)
    thread_01.start()
    time.sleep(5)
    # 创建线程2,云台控制
    thread_02 = Thread(target=myThread2)
    thread_02.start()
    time.sleep(5)
    while True:
        cv2.imshow("original", frame)
        if cv2.waitKey(1) == 27:
            break
    run(**vars(opt))


if __name__ == '__main__':
    opt = parse_opt()
    main(opt)
