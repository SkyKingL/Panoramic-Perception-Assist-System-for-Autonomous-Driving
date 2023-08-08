## 处理pred结果的.json文件,画图
import matplotlib.pyplot as plt
import cv2
import numpy as np
import random
from Common import Common

def plot_img_and_mask(img, mask, index,epoch,save_dir):
    classes = mask.shape[2] if len(mask.shape) > 2 else 1
    fig, ax = plt.subplots(1, classes + 1)
    ax[0].set_title('Input image')
    ax[0].imshow(img)
    if classes > 1:
        for i in range(classes):
            ax[i+1].set_title(f'Output mask (class {i+1})')
            ax[i+1].imshow(mask[:, :, i])
    else:
        ax[1].set_title(f'Output mask')
        ax[1].imshow(mask)
    plt.xticks([]), plt.yticks([])
    # plt.show()
    plt.savefig(save_dir+"/batch_{}_{}_seg.png".format(epoch,index))

def show_seg_result(img, result, index, epoch, save_dir=None, is_ll=False,palette=None,is_demo=True,is_gt=False):
    # img = mmcv.imread(img)
    # img = img.copy()
    # seg = result[0]

    if palette is None:
        # 返回一个3*3的随机向量，取值为[0,255]
        palette = np.random.randint(
                0, 255, size=(3, 3))
     # 下面就是调色板，根据预测结果可知，编号0代表背景，1代表车道，2代表车道线
    palette[0] = [0, 0, 0]
    palette[1] = [0, 255, 0]
    palette[2] = [255, 0, 0]
    palette = np.array(palette)
    assert palette.shape[0] == 3 # len(classes)，表示一共有3类，即二维数组有三行
    assert palette.shape[1] == 3 # 每一行对应rgb三列
    assert len(palette.shape) == 2 # 一共只有两个维度
    # 判断result是单个变量还是元组(da_mask,ll_mask)
    # is_demo == False->单个变量 
    # is_demo == True->元组
    # 如果是单个变量，就是可行驶区域划分结果或者车道线分割
    # 如果是元组，就是可行驶区域划分结果和车道线分割结果
    if not is_demo:  
        color_seg = np.zeros((result.shape[0], result.shape[1], 3), dtype=np.uint8)  # 最后一位3代表rgb三通道
        for label, color in enumerate(palette):
            color_seg[result == label, :] = color
    else: # result[0]表示车道,result[1]表示车道线。这里的result是元组
        color_area = np.zeros((result[0].shape[0], result[0].shape[1], 3), dtype=np.uint8)
        if Common.detect3 == True:
            color_area[result[0] == 1] = [0, 255, 0] # result[0]是车道掩码，网络会将预测的车道区域的像素置为1，保存在da_mask中，并显示为绿色。
        if Common.detect2 == True:
            color_area[result[1] ==1] = [255, 0, 0] # result[1]是车道线掩码，网络会将预测的车道线区域的像素置为1，保存在ll_mask中，并显示为红色。
        color_seg = color_area # 再次强调，是一个h*w*3的立方体

    # convert to BGR
    color_seg = color_seg[..., ::-1]
    # print(color_seg.shape)
    color_mask = np.mean(color_seg, 2) # axis == 2 ,表示对rgb取平均
    img[color_mask != 0] = img[color_mask != 0] * 0.5 + color_seg[color_mask != 0] * 0.5
    # img = img * 0.5 + color_seg * 0.5
    img = img.astype(np.uint8)

    # if not is_demo:
    #     if not is_gt:
    #         if not is_ll:
    #             cv2.imwrite(save_dir+"/batch_{}_{}_da_segresult.png".format(epoch,index), img)
    #         else:
    #             cv2.imwrite(save_dir+"/batch_{}_{}_ll_segresult.png".format(epoch,index), img)
    #     else:
    #         if not is_ll:
    #             cv2.imwrite(save_dir+"/batch_{}_{}_da_seg_gt.png".format(epoch,index), img)
    #         else:
    #             cv2.imwrite(save_dir+"/batch_{}_{}_ll_seg_gt.png".format(epoch,index), img)
    return img

def plot_one_box(x, img, color=None, label=None, line_thickness=None):
    # Plots one bounding box on image img
    tl = line_thickness or round(0.0001 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    # color是BGR格式
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    i = int((x[0]+x[2]) / 2) #midx
    j = int((x[1]+x[3]) / 2) #midy

    # 获取中心位置的深度数据 计算得距离
    # 深度数据的矩阵是(j,i) /256.0 是因为观察计算过程发现数据和真实数据之间 *256
    # *15是为了数据更真实 自己调

    # numpy获取矩阵的子矩阵
    out = Common.DepthOut[int(x[1]):int(x[3]), int(x[0]):int(x[2])]
    # numpy获取矩阵的最小值
    dist = round(np.min(out) / 256.0 * 15, 1)
    # dist = round(Common.DepthOut[j, i] / 256.0 * 15, 1)
    if Common.detect4:
        Common.label_det_pred = Common.label_det_pred + "  "+ str(dist) + "m"
    label = Common.label_det_pred
    # 显示label信息
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        print(label)
        # 图片 添加的文字 位置 字体 字体大小 字体颜色 字体粗细 行的类型
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [0, 0, 0], thickness=tf, lineType=cv2.LINE_AA)




if __name__ == "__main__":
    pass
# def plot():
#     cudnn.benchmark = cfg.CUDNN.BENCHMARK
#     torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
#     torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED

#     device = select_device(logger, batch_size=cfg.TRAIN.BATCH_SIZE_PER_GPU) if not cfg.DEBUG \
#         else select_device(logger, 'cpu')

#     if args.local_rank != -1:
#         assert torch.cuda.device_count() > args.local_rank
#         torch.cuda.set_device(args.local_rank)
#         device = torch.device('cuda', args.local_rank)
#         dist.init_process_group(backend='nccl', init_method='env://')  # distributed backend
    
#     model = get_net(cfg).to(device)
#     model_file = '/home/zwt/DaChuang/weights/epoch--2.pth'
#     checkpoint = torch.load(model_file)
#     model.load_state_dict(checkpoint['state_dict'])
#     if rank == -1 and torch.cuda.device_count() > 1:
#         model = torch.nn.DataParallel(model, device_ids=cfg.GPUS).cuda()
#     if rank != -1:
#         model = DDP(model, device_ids=[args.local_rank], output_device=args.local_rank)