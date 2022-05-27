# QQ2716490456
# 随风而息
# ————————————————
import os
import win32api
import win32gui
import win32con
import mss
import cv2
import numpy as np
from utils.general import non_max_suppression, scale_coords, xyxy2xywh
from utils.datasets import letterbox
import argparse
import torch
from models.experimental import attempt_load

'''
注意事项：1 . 选择识别模式后，需要修改路径的话(单个视频的路径修改就去单个视频识别，多个视频就去多个视频识别，屏幕识别就去屏幕识别)。
            路径前的r是屏蔽转义符，
        2 . 请检查模型路径'--weights',和所选的模式,及该模式的参数   !!!!!!!!!!
'''
parser = argparse.ArgumentParser()
parser.add_argument('--weights', type=str, default=r'C:\Users\Zzzz\Desktop\yolov5-6.0\weights\cf_410_epoch.pt',
                    help='模型地址,请使用绝对路径')
# parser.add_argument('--imgsz', type=int, default=640, help='和你训练模型时imgsz一样')
parser.add_argument('--cpu2cuda', type=bool, default=False, help='是否使用cuda')
parser.add_argument('--conf_thres', type=float, default=0.6, help='置信阈值')
parser.add_argument('--iou_thres', type=float, default=0.05, help='交并比阈值')

'''
                     ！！！！！！！！！！！！！！ 模式选择(必选)！！！！！！！！！！！！！
'''
parser.add_argument('--choose', type=int, default=2, help='模式选择:  0：单个视频识别，  1：多个视频识别，  2：屏幕识别')
parser.add_argument('--save_img', type=bool, default=False, help='是否保存检测到的图片')

'''
0：单个视频识别参数
'''
parser.add_argument('--video_path', type=str, default=r'D:\obs\video\297510698-1-80.flv', help='输入路径：单个<视频文件>的绝对路径')
parser.add_argument('--output_path', type=str, default=r'D:/obs/xml', help='输出路径单个视频文件的图片输出路径')
'''
1：多个视频识别参数
'''
parser.add_argument('--videos_path', type=str, default=r'D:\obs\xml', help='输入路径：多个视频<文件夹>的绝对路径')
parser.add_argument('--outputs_path', type=str, default=r'D:/obs/img', help='输出路径：多个视频文件的图片输出路径')
'''
2：屏幕识别参数
'''
parser.add_argument('--region', type=tuple, default=(1.0, 1.0),
                    help='屏幕检测范围；分别为横向和竖向，(1.0, 1.0)表示全屏检测，越低检测范围越小(始终保持屏幕中心为中心)')
parser.add_argument('--show-window', type=bool, default=True, help='是否显示实时检测窗口(新版里改进了效率。若为True，不要去点右上角的X！)')
parser.add_argument('--resize-window', type=float, default=1 / 2, help='缩放实时检测窗口大小')
parser.add_argument('--top-most', type=bool, default=True, help='是否保持实时检测窗口置顶，')
parser.add_argument('--mss_output_path', type=str, default=r'D:/obs/img', help='输出路径屏幕识别的图片输出路径')
'''
3.勿动,变量定义区域
'''
parser.add_argument('--img_size', type=int, default=640, help='')

args = parser.parse_args()

'''mss截图'''
sct = mss.mss()


def grab_screen_mss(monitor):
    try:
        return cv2.cvtColor(np.array(sct.grab(monitor)), cv2.COLOR_BGRA2BGR)
    except SyntaxError as e:
        print('错误1')


def get_parameters():
    try:
        x, y = get_screen_size().values()
        return 0, 0, x, y
    except SyntaxError as e:
        print('错误2')


def get_screen_size():
    try:
        wide = win32api.GetSystemMetrics(0)
        high = win32api.GetSystemMetrics(1)
        return {"wide": wide, "high": high}
    except SyntaxError as e:
        print('错误3')


'''model'''


def load_model(args):
    device = torch.device('cpu')  # 定义device 选择用gpu还是cpu
    # weights = r'C:\Users\Banditek\Desktop\yolov5-6.0\weights\CF-600img.pt'  # 训练好的模型地址
    imgsz = 640  # 训练时的分辨率是多少，这里就填多少

    model = attempt_load(args.weights, map_location=device)  # 导入模型

    if args.cpu2cuda:
        model.half()  # to FP16
    model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # 模型在cuda上初始化
    return model


''''''


def wangluo(image, aims):
    '''进行推理'''
    # 1.6ms
    img = letterbox(image, img_size, stride=stride)[0]  # 将截图image传入处理程序
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR 转 RGB, to 3x416x416    对图片进行格式转换
    img = np.ascontiguousarray(img)

    # 0.4ms
    img = torch.from_numpy(img).to(device)  # 4跟随推理程序的64行
    img = img.half() if half else img.float()  # 4
    img /= 255.0  # 4跟随66，对3个rgb（0.256）进行规划，对图片进行归化处理
    if img.ndimension() == 3:  # 4
        img = img[None]  # 4与img = img.unsqueeze(0)一模一样

    # 截图和对图片进行归化用了50-70ms左右
    '''----调用yolov5推理----'''
    # 50ms左右

    pred = model(img, augment='store_true')[0]  # 初始化数据.

    pred = non_max_suppression(pred, conf_thres=args.conf_thres, iou_thres=args.iou_thres, classes=None,
                               agnostic=False)  # 4跟随75，传入参数进行推理
    # tis2 = time.perf_counter()
    # print(tis2 - tis1)
    # print(pred)                                                     #输出检测到的图片数据

    '''----对推理的结果进行解密----'''
    # 2ms
    aims = []  # 定义一个变量列表，用于循环清空
    for i, det in enumerate(pred):  # 此for式对推理出来的图片进行解密
        gn = torch.tensor(image.shape)[[1, 0, 1, 0]]  # 用来归一化坐标,这里不需要归一化
        if len(det):
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], image.shape).round()
            for *xyxy, conf, cls in reversed(det):
                xywh = xyxy2xywh(torch.tensor(xyxy).view(1, 4)).view(-1).tolist()  # normalized xywh
                line = (cls, *xywh)  # 添加标签
                aim = ('%g ' * len(line)).rstrip() % line
                # print(aim)
                aim = aim.split(' ')
                print(aim, type(aim))
                # aims.append(aim)  # 将标签数据写入aims列表
    return aims


'''视频识别'''


def video2image(video_input, output_path):
    count = 0  # 计数用，分割的图片按照count来命名
    # 提取视频图片
    times = 0  # 用来记录帧
    frame_frequency = 10  # 提取视频的频率，每frameFrequency帧提取一张图片，提取完整视频帧设置为1

    cap = cv2.VideoCapture(video_input)  # 读取视频文件
    frameall = int(cap.get(7))  # 7获取总帧数,cv2.CAP_PROP_FRAME_COUNT
    print('本次识别视频一共有', frameall, '帧')
    print('开始提取', video_input, '视频')

    while True:
        aims = []
        times += 1
        res, image = cap.read()  # 读出图片。res表示是否读取到图片，image表示读取到的每一帧图片，每调用一次读取下一帧数
        if times >= frameall:
            print('————视频图片已提取结束')
            break
        '''进行推理'''
        img = letterbox(image, img_size, stride=stride)[0]  # 将截图image传入处理程序
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR 转 RGB, to 3x416x416    对图片进行格式转换
        img = np.ascontiguousarray(img)

        # 0.4ms
        img = torch.from_numpy(img).to(device)  # 4跟随推理程序的64行
        img = img.half() if half else img.float()  # 4
        img /= 255.0  # 4跟随66，对3个rgb（0.256）进行规划，对图片进行归化处理
        if img.ndimension() == 3:  # 4
            img = img[None]  # 4与img = img.unsqueeze(0)一模一样

            # 截图和对图片进行归化用了50-70ms左右
        '''----调用yolov5推理----'''
        # 50ms左右

        pred = model(img, augment='store_true')[0]  # 初始化数据.

        pred = non_max_suppression(pred, conf_thres=args.conf_thres, iou_thres=args.iou_thres, classes=None,
                                   agnostic=False)  # 4跟随75，传入参数进行推理
        '''推理结束，判断是否有识别出的数据'''
        if args.save_img:
            if len(pred):
                if times % frame_frequency == 0:
                    img_name = str(count).zfill(6) + '.png'  # 图片计数，6位上限
                    cv2.imwrite(output_path + os.sep + img_name, image)  # 存储图片
                    count += 1
                    # 输出提示
            if times % 100 == 0:
                # print(output_path + os.sep + img_name)
                print('\r视频提取进度:{:.1f}'.format((times / frameall) * 100), end='')

    cap.release()


def fun_mss(mss_output_path):
    '''循环函数'''
    count = 0  # 计数用，分割的图片按照count来命名
    cv2.namedWindow('detect', cv2.WINDOW_NORMAL)  # 创建窗口
    cv2.resizeWindow('detect', int(len_x * args.resize_window), int(len_y * args.resize_window))  # 定义窗口大小
    img_run = cv2.imread('../img/photo_20200624163252.png')
    cv2.imshow('detect', img_run)  # 显示窗口
    while True:
        if not cv2.getWindowProperty('detect', cv2.WND_PROP_VISIBLE):
            cv2.destroyAllWindows()
            exit('程序结束...')
            break
        aims = []
        image = grab_screen_mss(monitor)
        image = cv2.resize(image, (len_x, len_y))
        wangluo(image, aims)
        aims = aims
        if args.save_img:
            if len(aims):
                img_name = str(count).zfill(6) + '.jpg'  # 图片计数，6位上限
                cv2.imwrite(mss_output_path + os.sep + img_name, image)  # 存储图片
                count += 1
                print('已识别图片{} 并存入文件夹'.format(img_name), '————{ 在显示窗口按< L >键退出识别 or 直接关闭窗口 }————')
        if args.show_window:
            for i, det in enumerate(aims):
                tag, x_center, y_center, width, height = det
                x_center, width = len_x * float(x_center), len_x * float(width)
                y_center, height = len_y * float(y_center), len_y * float(height)
                top_left = (int(x_center - width / 2.), int(y_center - height / 2.))
                bottom_right = (int(x_center + width / 2.), int(y_center + height / 2.))
                cv2.rectangle(image, top_left, bottom_right, (0, 255, 0), thickness=3)
        if args.show_window:
            cv2.imshow('detect', image)  # 显示窗口
            if args.top_most:
                hwnd = win32gui.FindWindow(None, 'detect')
                CVRECT = cv2.getWindowImageRect('detect')
                win32gui.SetWindowPos(hwnd, win32con.HWND_TOPMOST, 0, 0, 0, 0,
                                      win32con.SWP_NOMOVE | win32con.SWP_NOSIZE)
        if cv2.waitKey(1) & 0xff == ord('l'):  # 2如果检测到按到‘l’，就退出此窗口，l可以自定义
            cv2.destroyAllWindows()
            break


top_x, top_y, x, y = get_parameters()  # x， y截取显示器的分辨率大小
len_x, len_y = int(x * args.region[0]), int(y * args.region[1])  # 原生分辨率*检测比例，横向对比#截图范围的左上角坐标
top_x, top_y = int(top_x + x // 2 * (1. - args.region[0])), int(
    top_y + y // 2 * (1. - args.region[1]))  ##截图范围的右下角坐标
monitor = {'left': top_x, 'top': top_y, 'width': len_x,
           'height': len_y}  # 用mss截取检测图片的分辨率大小，横向对比top_x = 0,  top_y = 0，len_x = 1920,  len_y = 1080

'''以上函数的变量定义'''
img_size = 640
model = load_model(args)
stride = int(model.stride.max())
device = 'cuda' if torch.cuda.is_available() else 'cpu'  # 4定义device
half = device != 'cpu'

if __name__ == '__main__':

    '''单个视频文件逐帧推理截图'''
    if args.choose == 0:
        # 视频路径   # 图片输出路径
        video_input = args.video_path  # r'D:\obs\apex\2022-01-11 21-25-45.mp4'
        output_path = args.output_path  # 'D:/obs/img'
        # 输出文件夹不存在，则创建输出文件夹
        if not os.path.exists(output_path):  # 不存在则创建输出文件夹
            os.makedirs(output_path)
        print('模式 0:———— 单个视频提取')
        video2image(video_input, output_path)  # run
        print('视频已提取完成，程序退出...')

        '''多个视频文件逐帧推理截图'''
    elif args.choose == 1:
        # 多个视频的文件夹
        videos_path = args.videos_path
        outputs_path = args.outputs_path
        lst1 = os.walk(videos_path)  # 查询该文件夹下的所有文件
        if not os.path.exists(outputs_path):
            os.makedirs(outputs_path)
        print('模式 1:———— 多个视频提取')
        for i in lst1:  # 遍历lst1所查询到的文件，i得到的是元组类型，有3个数据
            lst = i[2]  # 获取元组中的索引为2的数据 0:是主目录 ，1：是子目录，若没有主目录则返回空列表， 2：是所有目录下的文件
            print('-' * 100)
            print('此文件夹搜索到以下视频文件：', lst)
            print('-' * 100)
            for file in lst:  # 遍历lst的文件信息
                videos_input = os.path.join(videos_path, file)  # 将主目录和文件拼接，若有子目录则在videos_path后面添加子目录信息
                video2image(videos_input, outputs_path)  # run
        print('视频已提取完成，程序退出...')

        '''mss屏幕识别'''
    elif args.choose == 2:
        mss_output_path = args.mss_output_path  # 输出图片目录
        print('模式 2：———— 屏幕识别，在显示窗口按< L >键退出识别 or 直接关闭窗口')
        fun_mss(mss_output_path)
        print('等待程序退出...')
