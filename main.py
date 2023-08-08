import sys
from PySide2.QtWidgets import QApplication, QMainWindow, QMessageBox
from main_ui import Ui_MainWindow
from PySide2.QtWidgets import QFileDialog
from PySide2.QtGui import QImage, QPixmap, QIcon

import sys, os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
print(sys.path)
from PIL import Image
from Common import Common
from DepthDetect import DepthDetect

import argparse
import os, sys
import shutil
import time
from pathlib import Path
import imageio
import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
import scipy.special
import numpy as np
import torchvision.transforms as transforms
from PIL import Image

from PySide2.QtGui import QImage, QPixmap

from lib.config import cfg
from lib.config import update_config
from lib.utils.utils import create_logger, select_device, time_synchronized
from lib.models import get_net
from lib.dataset import LoadImages, LoadStreams
from lib.core.general import non_max_suppression, scale_coords
from lib.utils import plot_one_box,show_seg_result
from lib.core.function import AverageMeter
from lib.core.postprocess import morphological_process, connect_lane
from tqdm import tqdm

normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )

transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])

class MainWindow(QMainWindow):
    # 日志信息
    logger, _, _ = create_logger(
        cfg, cfg.LOG_DIR, 'demo')
    vid_path, vid_writer = None, None  # for video

    def __init__(self):
        # 基本初始化
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.setWindowTitle("自动驾驶全景感知辅助系统")
        # 相当于一个全局变量，用于后面可以暂停正在执行的事件
        # self.pause_event = Event()
        # self.pause_event.clear()

        # 给需要使用的控件声明好所需函数
        self.ui.Init.clicked.connect(self.InitDetect)  # 点击Init加载空白图片
        self.ui.select_img.clicked.connect(self.ImageDetect)
        self.ui.select_video.clicked.connect(self.VideoDetect)
        self.ui.pause.clicked.connect(self.Play)
        self.ui.rgbButton.clicked.connect(self.RGB)
        self.ui.exit.clicked.connect(self.exit)

        # 置信度、IoU
        self.ui.conf_SpinBox.valueChanged.connect(lambda x: self.change_val(x, 'confSpinBox'))
        self.ui.conf_Slider.valueChanged.connect(lambda x: self.change_val(x, 'confSlider'))
        self.ui.nms_SpinBoX.valueChanged.connect(lambda x: self.change_val(x, 'iouSpinBox'))
        self.ui.nms_Slider.valueChanged.connect(lambda x: self.change_val(x, 'iouSlider'))

        # 初始化设置复选框都为True
        self.ui.checkBox.setChecked(True)
        self.ui.checkBox_2.setChecked(True)
        self.ui.checkBox_3.setChecked(True)
        self.ui.checkBox_4.setChecked(True)
        self.ui.checkBox_5.setChecked(True)
        self.ui.checkBox_6.setChecked(True)
        self.ui.checkBox_8.setChecked(False)

        # 设置保存路径
        # self.ui.imgpath.setText(os.getcwd() + '\inference\output\image')
        # self.ui.videopath.setText(os.getcwd() + '\inference\output\\video')

        # D:\DaChuang\AutoDrive\inference\output\image
        self.ui.imgpath.setText("D:\DaChuang\AutoDrive\inference\output\image")
        self.ui.videopath.setText("D:\DaChuang\AutoDrive\inference\output\\video")

        # 为checkbox设置函数
        self.ui.checkBox.stateChanged.connect(lambda: self.change_checkBox(self.ui.checkBox))
        self.ui.checkBox_2.stateChanged.connect(lambda: self.change_checkBox(self.ui.checkBox_2))
        self.ui.checkBox_3.stateChanged.connect(lambda: self.change_checkBox(self.ui.checkBox_3))
        self.ui.checkBox_4.stateChanged.connect(lambda: self.change_checkBox(self.ui.checkBox_4))
        self.ui.checkBox_5.stateChanged.connect(lambda: self.change_checkBox(self.ui.checkBox_5))
        self.ui.checkBox_6.stateChanged.connect(lambda: self.change_checkBox(self.ui.checkBox_6))
        self.ui.checkBox_8.stateChanged.connect(lambda: self.change_checkBox(self.ui.checkBox_8))

        self.ui.imgsave.clicked.connect(self.save_img)
        self.ui.videosave.clicked.connect(self.save_video)

        parser = argparse.ArgumentParser()
        parser.add_argument('--weights', nargs='+', type=str, default='weights/End-to-end.pth',
                            help='model.pth path(s)')
        parser.add_argument('--source', type=str, default='', help='source')  # file/folder   ex:inference/images
        parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
        parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
        parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
        parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
        parser.add_argument('--save-dir', type=str, default='inference/output', help='directory to save results')
        parser.add_argument('--augment', action='store_true', help='augmented inference')
        parser.add_argument('--update', action='store_true', help='update all models')
        self.opt = parser.parse_args()

        # 设置device
        self.device = select_device(self.logger, self.opt.device)

        # 文件路径设置
        # if os.path.exists(self.opt.save_dir):  # output dir
        #     shutil.rmtree(self.opt.save_dir)  # delete dir
        # os.makedirs(self.opt.save_dir)  # make new dir
        self.half = self.device.type != 'cpu'  # half precision only supported on CUDA

        self.model = get_net(cfg)

    def InitDetect(self):
        # TODO
        # 通过一张空白照片，初始化并使用模型，有利于后续检测的性能
        self.LoadModel()
        self.Depth = DepthDetect()
        Common.savevid = False # 初始化的空白视频不用保存
        self.interface('blank.mp4')
        Common.savevid = True
        # self.pause_event.clear()

    def ImageDetect(self):
        filePath, _ = QFileDialog.getOpenFileName(
            self,  # 父窗口对象
            "选择你要上传的图片",  # 标题
            r"data//images",  # 起始目录
            "图片类型 (*.png *.jpg *.bmp  *.jpeg)"  # 选择类型过滤项，过滤内容在括号中
        )
        # self.pause_event.clear()
        Common.vidchange = True
        Common.rgbChange = False
        self.interface(filePath)

    def VideoDetect(self):
        filePath, _ = QFileDialog.getOpenFileName(
            self,  # 父窗口对象
            "选择你要上传的视频",  # 标题
            r"data//videos",  # 起始目录
            "视频类型 (*.mp4 *.*)"  # 选择类型过滤项，过滤内容在括号中 #*.*所有文件
        )

        # TODO
        # self.pause_event.clear()
        Common.rgbChange = False
        Common.vidchange = True
        Common.vidcount = Common.vidcount + 1
        self.interface(filePath)

    def Play(self):
        Common.play = not Common.play

    def exit(self):
        msgBox = QMessageBox()
        msgBox.setWindowTitle("退出")
        msgBox.setText("确定要关闭软件吗？")
        msgBox.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
        msgBox.setDefaultButton(QMessageBox.No) #默认是No
        ret = msgBox.exec_()
        if ret == QMessageBox.Yes:
            self.vid_writer.release()
            # 一秒时间执行 vid_writer.release()
            time.sleep(1)
            # 关闭窗口
            sys.exit()
        else:
            # cancel was clicked
            print('取消')

    def change_val(self, x, flag):
        if flag == 'confSpinBox':
            self.ui.conf_Slider.setValue(int(x * 100))
            self.opt.conf_thres = x / 100
        elif flag == 'confSlider':
            self.ui.conf_SpinBox.setValue(x / 100)
            self.opt.conf_thres = x / 100
        elif flag == 'iouSpinBox':
            self.ui.nms_Slider.setValue(int(x * 100))
            self.opt.iou_thres = x / 100
        elif flag == 'iouSlider':
            self.ui.nms_SpinBoX.setValue(x / 100)
            self.opt.iou_thres = x / 100
        else:
            pass

    def change_checkBox(self, x):
        if x == self.ui.checkBox:
            Common.saveimg = x.isChecked()
            print(Common.saveimg)
        elif x == self.ui.checkBox_2:
            Common.savevid = x.isChecked()
        elif x == self.ui.checkBox_3:
            Common.detect1 = x.isChecked()
        elif x == self.ui.checkBox_4:
            Common.detect2 = x.isChecked()
        elif x == self.ui.checkBox_5:
            Common.detect3 = x.isChecked()
        elif x == self.ui.checkBox_6:
            Common.detect4 = x.isChecked()
        elif x == self.ui.checkBox_8:
            Common.detect6 = x.isChecked()

    def RGB(self):
        Common.rgbChange = True

    def save_img(self):
        # 获取一个文件夹的路径

        dirPath = QFileDialog.getExistingDirectory(self,
                                    "选取文件夹",
                                    r'inference') # 起始路径
        if dirPath == "":
            print("\n取消选择")
            return
        self.ui.imgpath.setText(dirPath)


    def save_video(self):
        # 获取一个文件夹的路径

        dirPath = QFileDialog.getExistingDirectory(self,
                                    "选取文件夹",
                                    r'inference') # 起始路径
        if dirPath == "":
            print("\n取消选择")
            return
        self.ui.videopath.setText(dirPath)

    def LoadModel(self):
        # Load model 加载模型
        checkpoint = torch.load(self.opt.weights, map_location=self.device)
        self.model.load_state_dict(checkpoint['state_dict'])
        self.model = self.model.to(self.device)
        if self.half:
            self.model.half()  # to FP16
        s = 'Load Model ' + str(self.opt.weights) + ' success!'
        print(s)

    # 推理
    @torch.no_grad()
    def interface(self, filepath):
        self.opt.source = filepath
        # Set Dataloader 设置数据加载器
        if self.opt.source.isnumeric():
            cudnn.benchmark = True  # set True to speed up constant image size inference 设置为True可以加速固定图像大小的推理
            dataset = LoadStreams(self.opt.source, img_size=self.opt.img_size)
            bs = len(dataset)  # batch_size 批量大小
        else:
            dataset = LoadImages(self.opt.source, img_size=self.opt.img_size)
            bs = 1  # batch_size

        # Get names and colors 获取类别名称和颜色
        names = self.model.module.names if hasattr(self.model, 'module') else self.model.names
        colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

        # names是int数组，我定义的strnames是对应的类名
        strnames = ['car']

        # Run inference 运行推理
        t0 = time.time()

        img = torch.zeros((1, 3, self.opt.img_size, self.opt.img_size), device=self.device)  # init img 初始化图像
        _ = self.model(img.half() if self.half else img) if self.device.type != 'cpu' else None  # run once 仅运行一次
        self.model.eval()  # 设置为评估模式

        if Common.vidchange and Common.vidcount > 1:
            self.vid_writer.release()

        inf_time = AverageMeter()  # 推理时间
        nms_time = AverageMeter()  # nms时间

        tt1 = time.time()
        # 读取数据
        for i, (path, img, img_det, vid_cap, shapes) in tqdm(enumerate(dataset), total=len(dataset)):  # for each image
            if not Common.play:
                while not Common.play:
                    cv2.waitKey(1)
                    self.ui.pause.setText("播放")
            self.ui.pause.setText("暂停")
            img = transform(img).to(self.device)  # 图像预处理
            img = img.half() if self.half else img.float()  # uint8 to fp16/32 图像类型转换
            if img.ndimension() == 3:  # 图像维度转换
                img = img.unsqueeze(0)
            # Inference 推理
            t1 = time_synchronized()  # 获取当前时间
            det_out, da_seg_out, ll_seg_out = self.model(img)  # inference and training outputs 推理和训练输出
            # 检测深度，将深度数据放在图片中 需要对推理检测后的图片作为输入
            # 因为后期的注释中图片尺寸会受到影响

            print("ll_seg_out:" + str(ll_seg_out))
            
            t2 = time_synchronized()  # 获取当前时间
            # if i == 0:
            #     print(det_out)
            inf_out, _ = det_out  # inference output 推理输出
            inf_time.update(t2 - t1, img.size(0))  # 更新推理时间

            # Apply NMS 应用NMS
            t3 = time_synchronized()  # 获取当前时间
            det_pred = non_max_suppression(inf_out, conf_thres=self.opt.conf_thres, iou_thres=self.opt.iou_thres,
                                           classes=None, agnostic=False)
            t4 = time_synchronized()
            nms_time.update(t4 - t3, img.size(0))
            det = det_pred[0]

            # 车道线检测、可行驶区域划分的预处理
            _, _, height, width = img.shape
            h, w, _ = img_det.shape
            pad_w, pad_h = shapes[1][1]
            pad_w = int(pad_w)
            pad_h = int(pad_h)
            ratio = shapes[1][0][1]

            da_predict = da_seg_out[:, :, pad_h:(height - pad_h), pad_w:(width - pad_w)]
            da_seg_mask = torch.nn.functional.interpolate(da_predict, scale_factor=int(1 / ratio), mode='bilinear')
            _, da_seg_mask = torch.max(da_seg_mask, 1)
            da_seg_mask = da_seg_mask.int().squeeze().cpu().numpy()
            # da_seg_mask = morphological_process(da_seg_mask, kernel_size=7)
            ll_predict = ll_seg_out[:, :, pad_h:(height - pad_h), pad_w:(width - pad_w)]
            ll_seg_mask = torch.nn.functional.interpolate(ll_predict, scale_factor=int(1 / ratio), mode='bilinear')
            _, ll_seg_mask = torch.max(ll_seg_mask, 1)
            ll_seg_mask = ll_seg_mask.int().squeeze().cpu().numpy()
            # Lane line post-processing
            # ll_seg_mask = morphological_process(ll_seg_mask, kernel_size=7, func_type=cv2.MORPH_OPEN)
            # ll_seg_mask = connect_lane(ll_seg_mask)

            img_det = cv2.resize(img_det, (da_seg_mask.shape[1], da_seg_mask.shape[0]), interpolation=cv2.INTER_AREA)
            # # 车道线检测、可行驶区域划分 的显示
            img_det = show_seg_result(img_det, (da_seg_mask, ll_seg_mask), _, _, is_demo=True)

            # 检测深度，将深度数据放在图片中 需要对推理检测后的图片作为输入
            # 因为后期的注释中图片尺寸会受到影响
            im = Image.fromarray(img_det)
            Common.w = im.width
            Common.h = im.height
            self.Depth.detect(im)

            # 目标检测 的显示
            if len(det) and Common.detect1:
                # scale_coords 将检测结果转换到原图尺寸
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img_det.shape).round()
                for *xyxy, conf, cls in reversed(det):
                    idx = int(names[int(cls)])
                    Common.conf = conf
                    Common.names = names
                    if Common.rgbChange:
                        colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]
                        Common.rgbChange = False
                    Common.colors = colors
                    Common.label_det_pred = f'{strnames[idx]} {conf:.2f}' # f'{strnames[idx]} {conf:.2f}'
                    # plot_one_box(xyxy, img_det, label=Common.label_det_pred, color=colors[int(cls)], line_thickness=2)
                    plot_one_box(xyxy, img_det, label=Common.label_det_pred, color=colors[int(cls)], line_thickness=2)
            # cv2.imshow('result',img_det)
            # 保存图片或视频
            if dataset.mode == 'images' and Common.saveimg:
                self.opt.save_dir = self.ui.imgpath.text()
                if not os.path.exists(self.opt.save_dir):  # output dir
                    # shutil.rmtree(self.opt.save_dir)  # delete dir
                    os.makedirs(self.opt.save_dir)  # make new dir
                save_path = str(self.opt.save_dir + '/' + Path(path).name)
                cv2.imwrite(save_path, img_det)
            # 注意：要关闭软件后才能打开保存的视频
            elif dataset.mode == 'video' and Common.savevid:
                self.opt.save_dir = self.ui.videopath.text()
                if not os.path.exists(self.opt.save_dir):  # output dir
                    # shutil.rmtree(self.opt.save_dir)  # delete dir
                    os.makedirs(self.opt.save_dir)  # make new dir
                save_path = str(self.opt.save_dir + '/' + Path(path).name)
                if self.vid_path != save_path:  # new videos
                    self.vid_path = save_path
                    if isinstance(self.vid_writer, cv2.VideoWriter):
                        self.vid_writer.release()  # release previous video writer
                    # fourcc = 'mp4v'  # output video codec
                    fourcc = cv2.VideoWriter_fourcc('M', 'P', '4', 'V')  # 编码器 生成mp4
                    fps = vid_cap.get(cv2.CAP_PROP_FPS)
                    h, w, _ = img_det.shape
                    self.vid_writer = cv2.VideoWriter(save_path, fourcc, fps, (w, h))
                self.vid_writer.write(img_det) # 写入
            # 设置RGB
            image = QImage(img_det, img_det.shape[1], img_det.shape[0], img_det.strides[0], QImage.Format_BGR888)
            self.ui.out_video.setPixmap(QPixmap.fromImage(image)) # 图像放入label
            self.ui.out_video.setScaledContents(1) # 图片自适应
            while dataset.mode == "images":
                cv2.waitKey(1)
            cv2.waitKey(1)

            # 显示帧率
            # 帧率计算：也就是每秒多少帧,假设目标检测网络处理1帧要0.02s，此时FPS就是1/0.02=50
            tt2 = time.time()
            # self.ui.fps_label.setText("FPS:{:.1f}".format(1. / (tt2 - tt1)))
            tt1 = tt2

        # s = 'Results saved to %s' % str(self.opt.save_dir + '/' + Path(path).name)
        # self.ui.msg.setText(s)
        # print('Results saved to %s' % str(self.opt.save_dir + '/' + Path(path).name))
        print('Done. (%.3fs)' % (time.time() - t0))
        print('inf : (%.4fs/frame)   nms : (%.4fs/frame)' % (inf_time.avg, nms_time.avg))

if __name__ == '__main__':
    app = QApplication(sys.argv)
    mainw = MainWindow()
    mainw.show()
    sys.exit(app.exec_())

