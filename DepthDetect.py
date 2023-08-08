import torch
import numpy as np
from LapDepth.model import LDRN
import glob
import torch.backends.cudnn as cudnn
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F
import matplotlib.pyplot as plt
import argparse
import os
import cv2
import imageio

from Common import Common

class DepthDetect:

    def __init__(self):
        parser = argparse.ArgumentParser(description='Laplacian Depth Residual Network training on KITTI',
                                         formatter_class=argparse.ArgumentDefaultsHelpFormatter)

        # Directory setting

        parser.add_argument('--model_dir', type=str,
                            default='LapDepth/LDRN_KITTI_ResNext101_pretrained_data.pkl')  # 预训练的模型
        parser.add_argument('--img_dir', type=str, default="LapDepth/example/I-80_Eastshore_Fwy.jpg")  # 要预测的图片
        parser.add_argument('--img_folder_dir', type=str, default=None)  # 要预测的图片文件夹

        # Dataloader setting
        parser.add_argument('--seed', default=0, type=int,
                            help='seed for random functions, and network initialization')  # ?

        # Model setting
        parser.add_argument('--encoder', type=str, default="ResNext101")  # backbone
        parser.add_argument('--pretrained', type=str, default="KITTI")  # ?
        parser.add_argument('--norm', type=str, default="BN")  # 标准化
        parser.add_argument('--n_Group', type=int, default=32)  # ?
        parser.add_argument('--reduction', type=int, default=16)  # ?
        parser.add_argument('--act', type=str, default="ReLU")  # 激活函数
        parser.add_argument('--max_depth', type=float, default=80.0, metavar='MaxVal',
                            help='max value of depth')  # ?深度的最大值
        parser.add_argument('--lv6', action='store_true', help='use lv6 Laplacian decoder')

        # GPU setting
        parser.add_argument('--cuda', action='store_true')
        parser.add_argument('--gpu_num', type=str, default="0,1,2,3", help='force available gpu index')
        parser.add_argument('--rank', type=int, help='node rank for distributed training', default=0)  # ?

        self.args = parser.parse_args()  # 获取全部参数
        # 如果输入图片和图片文件夹路径都为空，则报错
        assert (self.args.img_dir is not None) or (
                self.args.img_folder_dir is not None), "Expected name of input image file or folder"

        if self.args.cuda and torch.cuda.is_available():
            os.environ["CUDA_VISIBLE_DEVICES"] = self.args.gpu_num  # 设置环境变量
            cudnn.benchmark = True
            print('=> on CUDA')
        else:
            print('=> on CPU')

        if self.args.pretrained == 'KITTI':
            self.args.max_depth = 80.0
        elif self.args.pretrained == 'NYU':
            self.args.max_depth = 10.0

        print('=> loading model..')
        self.Model = LDRN(self.args)  # LDRN
        if self.args.cuda and torch.cuda.is_available():
            self.Model = self.Model.cuda()
        self.Model = torch.nn.DataParallel(self.Model)  # 实现多GPU训练
        assert (self.args.model_dir != ''), "Expected pretrained model directory"  # 如果没设置预训练模型，报错
        self.Model.load_state_dict(torch.load(self.args.model_dir, map_location='cpu'))  # 将预训练的参数权重加载到新的模型之中
        self.Model.eval()  # eval()函数没太懂

        if self.args.img_dir is not None:
            if self.args.img_dir[-1] == '/':  # 如果路径的最后一个字符是'/'，则去掉'/'
                self.args.img_dir = self.args.img_dir[:-1]
            self.img_list = [self.args.img_dir]  # 图片路径
            self.result_filelist = ['LapDepth/Out/out_' + self.args.img_dir.split('/')[-1]]  # 输出图片，名字叫out_图片名
        elif self.args.img_folder_dir is not None:
            if self.args.img_folder_dir[-1] == '/':
                self.args.img_folder_dir = self.args.img_folder_dir[:-1]
            png_img_list = glob.glob(self.args.img_folder_dir + '/*.png')
            jpg_img_list = glob.glob(self.args.img_folder_dir + '/*.jpg')
            self.img_list = png_img_list + jpg_img_list  # 取输入文件夹中的png和jpg文件
            self.img_list = sorted(self.img_list)  # 升序排列
            result_folder = 'LapDepth/Out/out_' + self.args.img_folder_dir.split('/')[-1]
            if not os.path.exists(result_folder):  # 输出路径不存在则创建
                os.makedirs(result_folder)
            self.result_filelist = []
            for file in self.img_list:  # 对于图片文件夹中的每张图片
                self.result_filename = result_folder + '/out_' + file.split('/')[-1]
                self.result_filelist.append(self.result_filename)



    def detect(self, frame):
        # result_filelist = []
        print("=> process..")
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 标准化
        for i, img_file in enumerate(self.img_list):
            img = frame
            img = np.asarray(img, dtype=np.float32) / 255.0  # 为什么要除以255.0 ?
            if img.ndim == 2:  # 如果img的维度为2
                img = np.expand_dims(img, 2)  # 在第2维上扩展，即shape变成(x1, x2, 1)
                img = np.repeat(img, 3, 2)  # img 在第2维(列)上重复3次shape变成(x1, x2, 3)
            img = img.transpose((2, 0, 1))  # 转置成为 (3, x1, x2)
            img = torch.from_numpy(img).float()
            img = normalize(img)
            if self.args.cuda and torch.cuda.is_available():
                img = img.cuda()

            _, org_h, org_w = img.shape

            # new height and width setting which can be divided by 16   为什么要能被16整除?
            img = img.unsqueeze(0)  # 在第0维增加一个维度，变成(1, 3, x1, x2)

            if self.args.pretrained == 'KITTI':
                new_h = 352
                new_w = org_w * (352.0 / org_h)
                new_w = int((new_w // 16) * 16)  # // 是除法结果向下取整
                # F.interpolate：采样，img的shape变成(1, 3, new_h, new_w)。1,3分别是batch_size和channel
                img = F.interpolate(img, (new_h, new_w), mode='bilinear')
            elif self.args.pretrained == 'NYU':
                new_h = 432
                new_w = org_w * (432.0 / org_h)
                new_w = int((new_w // 16) * 16)
                img = F.interpolate(img, (new_h, new_w), mode='bilinear')

            # depth prediction  深度预测
            # with torch.no_grad():
            #    _, out = Model(img)

            # 数据增强？
            img_flip = torch.flip(img, [3])  # 按第三维(列)翻转
            with torch.no_grad():
                _, out = self.Model(img)  # 输出
                _, out_flip = self.Model(img_flip)  # 翻转后的输出
                out_flip = torch.flip(out_flip, [3])  # 再翻转回来？
                out = 0.5 * (out + out_flip)

            if new_h > org_h:  # ?
                out = F.interpolate(out, (org_h, org_w), mode='bilinear')
            out = out[0, 0]  # ?

            if self.args.pretrained == 'KITTI':
                out = out[int(out.shape[0] * 0.18):, :]  # ?
                out = out * 256.0
            elif self.args.pretrained == 'NYU':
                out = out * 1000.0
            out = out.cpu().detach().numpy().astype(np.uint16)
            out = (out / out.max()) * 256.0 #此处的out就是深度信息
            # print(out)

            # 矩阵转图像
            im0 = Image.fromarray(out)

            self.result_filename = self.result_filelist[i]
            # plasma_r 彩
            # gray 灰
            plt.imsave(self.result_filename, np.log10(out), cmap='gray')  # 保存图像
            # if (i + 1) % 10 == 0:
            #     print("=>", i + 1, "th image is processed..")
            img = cv2.imread(self.result_filelist[i])
            if Common.detect6:
                cv2.imshow('Depth Image', img)
                cv2.waitKey(1)
            # img转三元矩阵
            im = Image.fromarray(img)
            # w1、h1是定值 在初始化时已获得
            Common.x = Common.w * 1.0 / im.width
            Common.y = Common.h * 1.0 / im.height
            im = im.resize((int(im.width * Common.x), int(im.height * Common.y)), Image.ANTIALIAS)
            out_ = im.convert("RGB")
            Common.dataset = np.array(out_)

            im0 = im0.resize((int(im0.width * Common.x), int(im0.height * Common.y)), Image.ANTIALIAS)
            # resize后图像转矩阵
            # resize的目的：
            # 深度检测后生成的图片大小和yolov5+yolopv2模型检测后生成的图片大小不一致
            # 要通过像素位置的映射判断某像素位置中的深度数据
            Common.DepthOut = np.array(im0)

            # waitkey控制对窗口进行操作
            cv2.waitKey(1)

