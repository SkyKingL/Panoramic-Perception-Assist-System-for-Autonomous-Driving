import numpy as np


class Common:
    img = 0
    saveimg = True
    savevid = True
    detect1 = True # 目标检测
    detect2 = True # 车道线检测
    detect3 = True # 可行驶区域划分
    detect4 = True # 前车距离估计
    # detect5 = True # 车道弯曲度
    detect6 = False # 深度图

    # 显示在QT中图片的宽和高
    w = 1
    h = 1
    DepthOut = 0
    # 图片宽和高的比例
    x = 1
    y = 1

    label_det_pred = ""
    names = []
    colors = []

    play = True

    vidchange = False

    vidcount = 0

    rgbChange = False

