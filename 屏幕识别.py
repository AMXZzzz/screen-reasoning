import mss
from models.experimental import attempt_load
import torch
import numpy as np
from utils.general import non_max_suppression, scale_coords, xyxy2xywh
from utils.augmentations import letterbox
import argparse
import time
import cv2  # opencv-python 不要超过4.5

'''
v6.0版本训练出来的模型必须和同版本神经网络一致，否则会报错
'''
parser = argparse.ArgumentParser()
parser.add_argument('--model-path', type=str, default=r'C:\Users\Zzzz\Desktop\yolov5-6.0\weights\cf_v6.pt',
                    help='模型地址,绝对路径')
parser.add_argument('--imgsz', type=int, default=640, help='和你训练模型时imgsz一样,默认640')
parser.add_argument('--conf-thres', type=float, default=0.1, help='置信阈值')
parser.add_argument('--iou-thres', type=float, default=0.05, help='交并比阈值')
parser.add_argument('--show-window', type=bool, default=True, help='是否显示实时检测窗口')
parser.add_argument('--resize-window', type=float, default=1 / 2, help='缩放实时检测窗口大小')
parser.add_argument('--show-fps', type=bool, default=True, help='是否显示帧数')
parser.add_argument('--region', type=tuple, default=(0.5, 0.5),
                    help='检测范围；分别为x，y，(1.0, 1.0)表示全屏检测，越低检测范围越小(始终保持屏幕中心为中心)')
args = parser.parse_args()
'------------------------------------------------------------------------------------'

# 加载模型
def load_model(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'  # 设备选择
    half = device != 'cpu'  # fp32/fp16
    model = attempt_load(args.model_path, map_location=device)  # 加载 FP32 模型
    if half:  # 如果cuda可用
        model.half()  # 启用 FP16

    if device != 'cpu':  # 如果是cuda
        model(torch.zeros(1, 3, args.imgsz, args.imgsz).to(device).type_as(next(model.parameters())))  # cuda设置
    return model  # 返回加载好的网络模型


# mss截图
cap = mss.mss()  # 实例化mss


def grab_screen_mss(monitor):
    # cap.grab截取图片，np.array将图片转为数组，cvtColor将BRGA转为BRG,去掉了透明通道
    return cv2.cvtColor(np.array(cap.grab(monitor)), cv2.COLOR_BGRA2BGR)


# 画框函数
def fun_en(aims, img0, len_x, len_y):
    for i, det in enumerate(aims):
        _, x_center, y_center, width, height = det  # 将det里的数据分装到前面   rc_x,y  表示归化后的比例坐标
        x_center, width = len_x * float(x_center), len_x * float(width)  # 中心的x和宽
        y_center, height = len_y * float(y_center), len_y * float(height)  # 中心的y和高
        top_left = (int(x_center - width / 2.), int(y_center - height / 2.))
        bottom_right = (int(x_center + width / 2.), int(y_center + height / 2.))
        color = (0, 255, 0)  # RGB     框的颜色
        cv2.rectangle(img0, top_left, bottom_right, color, thickness=3)  # 3代表线条粗细


# 运行
def run():
    top_x, top_y, x, y = 0, 0, 1920, 1080  # x,y 屏幕大小,top是原点
    len_x, len_y = int(x * args.region[0]), int(y * args.region[1])  # 截图的宽高
    top_x, top_y = int(top_x + x // 2 * (1. - args.region[0])), int(top_y + y // 2 * (1. - args.region[1]))  # 截图区域的原点
    monitor = {'left': top_x, 'top': top_y, 'width': len_x, 'height': len_y}  # 截图范围

    cv2.namedWindow('img', cv2.WINDOW_NORMAL)  # 创建窗口
    if args.show_window:  # 是否显示检测款
        len_x, len_y = int(x * args.region[0]), int(y * args.region[1])
        cv2.resizeWindow('img', int(len_x * args.resize_window), int(len_y * args.resize_window))  # 裁剪窗口

    t0 = time.time()  # fps 计算
    while True:
        if not cv2.getWindowProperty('img', cv2.WND_PROP_VISIBLE):  # 如果窗口关闭，退出程序
            cv2.destroyAllWindows()
            exit('程序结束...')
            break

        img0 = grab_screen_mss(monitor)  # 截取整个屏幕的到图片img0
        img0 = cv2.resize(img0, (len_x, len_y))  # 裁剪图片至截取的大小

        # 预处理
        img = letterbox(img0, args.imgsz, stride=stride)[0]  # 预处理
        img = img.transpose((2, 0, 1))[::-1]  # 维度转换
        img = np.ascontiguousarray(img)  # 转为数组，其内存是连续的
        img = torch.from_numpy(img).to(device)  # 将来自numpy的数组转为tensor，并传入设备
        img = img.half() if half else img.float()  # 选择fp32 / fp16
        img /= 255.  # 归一化
        img = img[None]  # 扩大批调暗
        # if len(img.shape) == 3:
        #     img = img[None]

        # 推理
        t1 = time.time()  # 时间点
        pred = model(img, augment=False, visualize=False)[0]

        # 非极大值抑制
        pred = non_max_suppression(pred, conf_thres, iou_thres, agnostic=False)
        t2 = time.time()

        print('推理时间 {} ms'.format('%.2f' % ((t2 - t1) * 1000)))

        # 转换
        aims = []
        for i, det in enumerate(pred):
            if len(det):
                # 将坐标 (xyxy) 从 img_shape 重新缩放为 img0_shape
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()
                for *xyxy, conf, cls in reversed(det):  # 从末尾遍历
                    # 将xyxy合并至一个维度,锚框的左上角和右下角
                    xyxy = (torch.tensor(xyxy).view(1, 4)).view(-1)
                    # 将类别和坐标合并
                    line = (cls, *xyxy)
                    # 提取tensor类型里的坐标数据
                    aim = ('%g ' * len(
                        line)).rstrip() % line  # %g 格式为浮点数 .rstrip()删除tring字符串末尾的指定字符,默认为空白符包括空格,即删除2个坐标之间的空格
                    # 划分元素
                    aim = aim.split(' ')  # 将一个元素按空格符分为多个元素,获得单个目标信息列表
                    # 所有目标的类别和锚框的坐标(类别,左上角x,左上角y,右下角x,右下角y)
                    aims.append(aim)  # 添加至列表
                    aims.append(aim)  # 加入标签列表

            if len(aims):  # 如果检测到存在目标
                fun_en(aims, img0, len_x, len_y)  # 画框函数

        # 显示检测
        if args.show_window:  # 是否显示窗口
            if args.show_fps:  # 是否显示 fps
                cv2.putText(img0, "FPS:{:.1f}".format(1. / (time.time() - t0)), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2,
                            (0, 0, 235), 4)  # 绘制字体
                t0 = time.time()
            cv2.imshow('img', img0)  # 显示
        cv2.waitKey(1)


if __name__ == '__main__':
    # 参数初始化
    model = load_model(args)  # 加载模型
    stride = int(model.stride.max())  # 设置特征点步长
    device = 'cuda' if torch.cuda.is_available() else 'cpu'  # 根据pytorch选择设备，cpu或者cuda
    conf_thres = args.conf_thres  # 置信度
    iou_thres = args.iou_thres  # IOU
    half = device != 'cpu'  # 如果cuda可用，启用fp16

    # run
    run()
