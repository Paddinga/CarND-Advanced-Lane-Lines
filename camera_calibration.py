import cv2
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


class CameraCalibration:
    ret = False
    mtx = []
    dist = []
    rvecs = 0
    tvecs = 0
    src = np.float32([
        (602, 445),
        (680, 445),
        (1040, 675),
        (275, 675)])
    dst = np.float32([
        (340, 0),
        (940, 0),
        (940, 720),
        (340, 720)])

    def __init__(self):
        print('# Calibration initialized ...')

    # Get camera calibration parameters from chessboard images
    def calibrate_camera(self):
        calib_images = glob.glob("camera_cal/calibration*.jpg")
        points = np.zeros((6 * 9, 3), np.float32)
        points[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)
        objpoints = []
        imgpoints = []
        for calib_image in calib_images:
            img = mpimg.imread(calib_image)
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, (9, 6), None)
            if ret == True:
                objpoints.append(points)
                imgpoints.append(corners)
                i = 0
        assert len(objpoints) == len(imgpoints)
        self.ret, self.mtx, self.dist, self.rvecs, self.tvecs = \
            cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
        print('# Camera calibrated with: {} images'.format(len(calib_images)))

    # Undistort images
    def undistort(self, image, plot=False):
        if not self.ret:
            print('# Calibrate camera first!')
            return 0
        else:
            image_undist = cv2.undistort(image, self.mtx, self.dist, None, self.mtx)
            if plot:
                fig, (img1, img2) = plt.subplots(1, 2)
                fig.canvas.set_window_title('Distorted and Undistorted Image')
                img1.imshow(image)
                img2.imshow(image_undist)
                plt.show(fig)
            return image_undist

    # Get warp parameter and warp/unwarp image
    def warp(self, image):
        M = cv2.getPerspectiveTransform(self.src, self.dst)
        image_size = (image.shape[1], image.shape[0])
        return cv2.warpPerspective(image, M, image_size, flags=cv2.INTER_LINEAR)

    def unwarp(self, image):
        M = cv2.getPerspectiveTransform(self.dst, self.src)
        image_size = (image.shape[1], image.shape[0])
        return cv2.warpPerspective(image, M, image_size, flags=cv2.INTER_LINEAR)
