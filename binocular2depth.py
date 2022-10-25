import torch
import torch.nn.functional as F
import numpy as np
import cv2
import os
import glob
import math

from tqdm import tqdm

from calibration import StereoCalibration, stereo
from nets import Model

device = 'cuda'



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
        left_K, left_distortion, R1, P1, (width, height), cv2.CV_32FC1) # 16SC2
    map2x, map2y = cv2.initUndistortRectifyMap(
        right_K, right_distortion, R2, P2, (width, height), cv2.CV_32FC1)
    # print(width, height)

    return map1x, map1y, map2x, map2y, Q


# 畸变校正和立体校正
def rectifyImage(image1, image2, map1x, map1y, map2x, map2y):
    rectifyed_img1 = cv2.remap(image1, map1x, map1y, cv2.INTER_LINEAR, cv2.BORDER_TRANSPARENT)
    rectifyed_img2 = cv2.remap(image2, map2x, map2y, cv2.INTER_LINEAR, cv2.BORDER_TRANSPARENT)

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

def inference(left, right, model, n_iter=20):

	# print("Model Forwarding...")
	imgL = left.transpose(2, 0, 1)
	imgR = right.transpose(2, 0, 1)
	imgL = np.ascontiguousarray(imgL[None, :, :, :])
	imgR = np.ascontiguousarray(imgR[None, :, :, :])

	imgL = torch.tensor(imgL.astype("float32")).to(device)
	imgR = torch.tensor(imgR.astype("float32")).to(device)

	imgL_dw2 = F.interpolate(
		imgL,
		size=(imgL.shape[2] // 2, imgL.shape[3] // 2),
		mode="bilinear",
		align_corners=True,
	)
	imgR_dw2 = F.interpolate(
		imgR,
		size=(imgL.shape[2] // 2, imgL.shape[3] // 2),
		mode="bilinear",
		align_corners=True,
	)
	# print(imgR_dw2.shape)
	with torch.no_grad():
		pred_flow_dw2 = model(imgL_dw2, imgR_dw2, iters=n_iter, flow_init=None)

		pred_flow = model(imgL, imgR, iters=n_iter, flow_init=pred_flow_dw2)
	pred_disp = torch.squeeze(pred_flow[:, 0, :, :]).cpu().detach().numpy()

	return pred_disp

def block_process(sid, eid):
    for idx in range(sid, eid):
        imgL = cv2.imread(imgL_dir[idx])
        imgR = cv2.imread(imgR_dir[idx])

        height, width = imgL.shape[0:2]
        
        # 去畸变
        imgL = undistortion(imgL, config.cam_matrix_left, config.distortion_l)
        imgR = undistortion(imgR, config.cam_matrix_right, config.distortion_r)
    
        # 去畸变和几何极线对齐
        map1x, map1y, map2x, map2y, Q = getRectifyTransform(height, width, config)
        imgL, imgR = rectifyImage(imgL, imgR, map1x, map1y, map2x, map2y)
        if draw_process:
            linepic = draw_line_RGB(imgL, imgR)
            cv2.imshow("rectify", linepic)
            # cv2.imshow("left.bmp", imgL)
            # cv2.imshow("right.bmp", imgR)
        
        # imgL = remove_black_borders(imgL)
        # imgR = remove_black_borders(imgR)
        # cv2.imshow("imgL", imgL)
        
        
        # Resize image in case the GPU memory overflows
        eval_h, eval_w = (height,width)
        assert eval_h % 8 == 0, "input height should be divisible by 8"
        assert eval_w % 8 == 0, "input width should be divisible by 8"

        imgL = cv2.resize(imgL, (eval_w, eval_h), interpolation=cv2.INTER_LINEAR)
        imgR = cv2.resize(imgR, (eval_w, eval_h), interpolation=cv2.INTER_LINEAR)


        pred = inference(imgL, imgR, model, n_iter=20)

        t = float(width) / float(eval_w)
        disp = cv2.resize(pred, (width, height), interpolation=cv2.INTER_LINEAR) * t

        disp_vis = (disp - disp.min()) / (disp.max() - disp.min()) * 255.0
        disp_vis = disp_vis.astype("uint8")
        disp_vis = cv2.applyColorMap(disp_vis, cv2.COLORMAP_INFERNO)

        path = out_dir + imgL_dir[idx].split("/")[-1]
        # combined_img = np.hstack((imgL, disp_vis))
        # cv2.namedWindow("output", cv2.WINDOW_NORMAL)
        # cv2.imshow("output", combined_img)
        cv2.imwrite(path, disp_vis)
        


def remove_black_borders(image):
    
    ret, image = cv2.threshold(image,5,255,cv2.THRESH_BINARY_INV)
    # print(image)
    y_nonzero, x_nonzero, _ = np.nonzero(image)
    return image[np.min(y_nonzero):np.max(y_nonzero), np.min(x_nonzero):np.max(x_nonzero)]

if __name__ == '__main__':

    stereo.show()
    if not type(stereo.m1) is np.ndarray:
        calibration = StereoCalibration()
        calibration.calibration_photo(11, 8)
        stereo.show()

    draw_process = False
    data_dir = "frames/22-10-26_01-13-42/"
    out_dir = "depth/22-10-26_01-13-42/"
    os.system("rm -rf {0} && mkdir {0}".format(out_dir))

    imgL_dir = sorted(glob.glob(data_dir+"left/*.bmp"))
    imgR_dir = sorted(glob.glob(data_dir+"right/*.bmp"))
    assert len(imgL_dir) == len(imgR_dir)

    config = stereoCameral(stereo)    # 读取相机内参和外参
    
    model_path = "models/crestereo_eth3d.pth"

    model = Model(max_disp=256, mixed_precision=False, test_mode=True)
    model.load_state_dict(torch.load(model_path), strict=True)
    model.to(device)
    model.eval()
    
    from joblib import Parallel, delayed
    import multiprocessing
    import subprocess
    block_size = 10
    MAX_THREAD = min(multiprocessing.cpu_count(), 15)
    Parallel(n_jobs=MAX_THREAD)(delayed(block_process)(
        sid*block_size, min((sid+1)*block_size, len(imgL_dir)))
        for sid in range(len(imgL_dir)//block_size))
    
    # for idx in tqdm(range(len(imgL_dir))):
    #     imgL = cv2.imread(imgL_dir[idx])
    #     imgR = cv2.imread(imgR_dir[idx])

    #     height, width = imgL.shape[0:2]
        
    #     # 去畸变
    #     imgL = undistortion(imgL, config.cam_matrix_left, config.distortion_l)
    #     imgR = undistortion(imgR, config.cam_matrix_right, config.distortion_r)
    #     if draw_process:
    #         linepic = draw_line_RGB(imgL, imgR)
    #         cv2.imshow("undistortion", linepic)

    #     # 去畸变和几何极线对齐
    #     map1x, map1y, map2x, map2y, Q = getRectifyTransform(height, width, config)
    #     imgL, imgR = rectifyImage(imgL, imgR, map1x, map1y, map2x, map2y)
    #     if draw_process:
    #         linepic = draw_line_RGB(imgL, imgR)
    #         cv2.imshow("rectify", linepic)
    #         # cv2.imshow("left.bmp", imgL)
    #         # cv2.imshow("right.bmp", imgR)
        
    #     # imgL = remove_black_borders(imgL)
    #     # imgR = remove_black_borders(imgR)
    #     # cv2.imshow("imgL", imgL)
        
        
    #     # Resize image in case the GPU memory overflows
    #     eval_h, eval_w = (height,width)
    #     assert eval_h % 8 == 0, "input height should be divisible by 8"
    #     assert eval_w % 8 == 0, "input width should be divisible by 8"

    #     imgL = cv2.resize(imgL, (eval_w, eval_h), interpolation=cv2.INTER_LINEAR)
    #     imgR = cv2.resize(imgR, (eval_w, eval_h), interpolation=cv2.INTER_LINEAR)


    #     pred = inference(imgL, imgR, model, n_iter=20)

    #     t = float(width) / float(eval_w)
    #     disp = cv2.resize(pred, (width, height), interpolation=cv2.INTER_LINEAR) * t

    #     disp_vis = (disp - disp.min()) / (disp.max() - disp.min()) * 255.0
    #     disp_vis = disp_vis.astype("uint8")
    #     disp_vis = cv2.applyColorMap(disp_vis, cv2.COLORMAP_INFERNO)

    #     path = out_dir + imgL_dir[idx].split("/")[-1]
    #     # combined_img = np.hstack((imgL, disp_vis))
    #     # cv2.namedWindow("output", cv2.WINDOW_NORMAL)
    #     # cv2.imshow("output", combined_img)
    #     cv2.imwrite(path, disp_vis)
    #     if draw_process:
    #         while True:
    #             if cv2.waitKey(0) & 0xFF == ord('q'):
    #                 break
        



