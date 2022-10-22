
import cv2
import math
import numpy as np
import matplotlib.pyplot as plt  # plt 用于显示图片
import matplotlib.image as mpimg  # mpimg 用于读取图片

from calibration import StereoCalibration, stereo

from highres_stereo import HighResStereo
from highres_stereo.utils_highres import Config, CameraConfig, draw_disparity, draw_depth, QualityLevel

# 双目相机参数


class stereoCameral(object):
    def __init__(self, stereo):
        # 左相机内参数
        self.cam_matrix_left = stereo.m1
        # 右相机内参数
        self.cam_matrix_right = stereo.m2

        #左右相机畸变系数:[k1, k2, p1, p2, k3]
        self.distortion_l = stereo.d1
        self.distortion_r = stereo.d2
        # 旋转矩阵

        self.R = stereo.R
        # 平移矩阵
        self.T = stereo.T

        self.baseline = stereo.T[0]


# 预处理
def preprocess(img1, img2):
    # 彩色图->灰度图
    im1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    im2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # 直方图均衡
    im1 = cv2.equalizeHist(im1)
    im2 = cv2.equalizeHist(im2)

    return im1, im2

# 消除畸变


def undistortion(image, camera_matrix, dist_coeff):
    undistortion_image = cv2.undistort(image, camera_matrix, dist_coeff)

    return undistortion_image

# 获取畸变校正和立体校正的映射变换矩阵、重投影矩阵
# @param：config是一个类，存储着双目标定的参数:config = stereoconfig.stereoCamera()


def getRectifyTransform(height, width, config):
    # 读取内参和外参
    left_K = config.cam_matrix_left
    right_K = config.cam_matrix_right
    left_distortion = config.distortion_l
    right_distortion = config.distortion_r
    R = config.R
    T = config.T

    # 计算校正变换
    height = int(height)
    width = int(width)
    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
        left_K, left_distortion, right_K, right_distortion, (width, height), R, T, alpha=0) # alpha = -1

    map1x, map1y = cv2.initUndistortRectifyMap(
        left_K, left_distortion, R1, P1, (width, height), cv2.CV_16SC2)
    map2x, map2y = cv2.initUndistortRectifyMap(
        right_K, right_distortion, R2, P2, (width, height), cv2.CV_16SC2)
    print(width, height)

    return map1x, map1y, map2x, map2y, Q


# 畸变校正和立体校正
def rectifyImage(image1, image2, map1x, map1y, map2x, map2y):
    rectifyed_img1 = cv2.remap(image1, map1x, map1y, cv2.INTER_LINEAR)
    rectifyed_img2 = cv2.remap(image2, map2x, map2y, cv2.INTER_LINEAR)

    return rectifyed_img1, rectifyed_img2

# 立体校正检验----画线


def draw_line_RGB(image1, image2):
    # 建立输出图像
    height = max(image1.shape[0], image2.shape[0])
    width = image1.shape[1] + image2.shape[1]

    output = np.zeros((height, width, 3), dtype=np.uint8)
    output[0:image1.shape[0], 0:image1.shape[1]] = image1
    output[0:image2.shape[0], image1.shape[1]:] = image2

    for k in range(15):
        cv2.line(output, (0, 50 * (k + 1)), (2 * width, 50 * (k + 1)),
                 (0, 255, 0), thickness=2, lineType=cv2.LINE_AA)  # 直线间隔：100
    return output
# 立体校正检验----画线


def draw_line_depth(image1, image2):
    # 建立输出图像
    height = max(image1.shape[0], image2.shape[0])
    width = image1.shape[1] + image2.shape[1]

    output = np.zeros((height, width), dtype=np.uint8)
    output[0:image1.shape[0], 0:image1.shape[1]] = image1
    output[0:image2.shape[0], image1.shape[1]:] = image2
    for k in range(15):
        cv2.line(output, (0, 50 * (k + 1)), (2 * width, 50 * (k + 1)),
                 (0, 255, 0), thickness=2, lineType=cv2.LINE_AA)  # 直线间隔：100
    return output


# 视差计算
def disparity_SGBM(left_image, right_image, down_scale=False):
    # SGBM匹配参数设置
    if left_image.ndim == 2:
        img_channels = 1
    else:
        img_channels = 3
    blockSize = 3  # 3
    param = {'minDisparity': 0,
             'numDisparities': 16,  # 128
             'blockSize': blockSize,
             'P1': 8 * img_channels * blockSize ** 2,
             'P2': 32 * img_channels * blockSize ** 2,
             'disp12MaxDiff': 1,
             'preFilterCap': 63,
             'uniquenessRatio': 15,
             'speckleWindowSize': 100,
             'speckleRange': 1,
             'mode': cv2.STEREO_SGBM_MODE_SGBM_3WAY
             }

    # 构建SGBM对象
    sgbm = cv2.StereoSGBM_create(**param)

    # 计算视差图
    size = (left_image.shape[1], left_image.shape[0])
    if down_scale == False:
        disparity_left = sgbm.compute(left_image, right_image)
        disparity_right = sgbm.compute(right_image, left_image)
    else:
        left_image_down = cv2.pyrDown(left_image)
        right_image_down = cv2.pyrDown(right_image)
        factor = size[0] / left_image_down.shape[1]
        disparity_left_half = sgbm.compute(left_image_down, right_image_down)
        disparity_right_half = sgbm.compute(right_image_down, left_image_down)
        disparity_left = cv2.resize(
            disparity_left_half, size, interpolation=cv2.INTER_AREA)
        disparity_right = cv2.resize(
            disparity_right_half, size, interpolation=cv2.INTER_AREA)
        disparity_left = disparity_left * factor
        disparity_right = disparity_right * factor

    return disparity_left, disparity_right


def disparity_BM(left_image, right_image, down_scale=False):
    stereo = cv2.StereoBM_create(numDisparities=16, blockSize=15)
    disparityL = stereo.compute(left_image, right_image)
    disparityR = stereo.compute(right_image, left_image)

    return disparityL, disparityR


if __name__ == '__main__':

    stereo.show()
    if not type(stereo.m1) is np.ndarray:
        calibration = StereoCalibration()
        calibration.calibration_photo(11, 8)
        stereo.show()

    data_dir = "frames/22-10-22_15-34-30/"
    imgL = cv2.imread(data_dir+"left/00022.png")
    imgR = cv2.imread(data_dir+"right/00022.png")
    #imgL , imgR = preprocess(imgL ,imgR )

    height, width = imgL.shape[0:2]
    config = stereoCameral(stereo)    # 读取相机内参和外参

    # 去畸变
    imgL = undistortion(imgL, config.cam_matrix_left, config.distortion_l)
    imgR = undistortion(imgR, config.cam_matrix_right, config.distortion_r)
    linepic = draw_line_RGB(imgL, imgR)
    cv2.imwrite("undistortion.jpg", linepic)

    # 去畸变和几何极线对齐
    map1x, map1y, map2x, map2y, Q = getRectifyTransform(height, width, config)
    imgL, imgR = rectifyImage(imgL, imgR, map1x, map1y, map2x, map2y)

    linepic = draw_line_RGB(imgL, imgR)
    cv2.imwrite("rectify.jpg", linepic)
    # cv2.imwrite("left.png", imgL)
    # cv2.imwrite("right.png", imgR)
    
    config = Config(clean=-1, qualityLevel=QualityLevel.High,
                    max_disp=128, img_res_scale=1)
    model_path = "models/final-768px.tar"
    # Initialize model
    highres_stereo_depth = HighResStereo(model_path, config, use_gpu=True)
    # Estimate the depth
    disparity_map = highres_stereo_depth(imgL, imgR)

    color_disparity = draw_disparity(disparity_map)
    print(color_disparity)
    color_disparity = cv2.resize(
        color_disparity, (imgL.shape[1], imgL.shape[0]))
    cv2.imwrite("depth.jpg", color_disparity)
    combined_image = np.hstack((imgL, imgR, color_disparity))
    combined_image = cv2.putText(combined_image, f'{highres_stereo_depth.fps} fps', (
        50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.imwrite("comparison.jpg", combined_image)
