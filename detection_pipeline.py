import cv2
import numpy as np
import matplotlib.pyplot as plt
from camera_calibration import CameraCalibration
from moviepy.editor import VideoFileClip


class LaneLineDetector():
    left_history = [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]]
    right_history = [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]]
    left_mean = [0, 0, 0]
    right_mean = [0, 0, 0]
    # Region of Interest
    VERTICES = np.array([[
        (220, 650),                         #bottom-left
        (540, 450),                         #top-left
        (740, 450),                         #top-right
        (1060, 650)]], dtype=np.int32)      #bottom-right
    # Parameters for image corretion
    KERNEL_SOBEL = 9
    THRESH_CORR_RED = (200, 255)
    THRESH_CORR_SAT = (100, 255)
    THRESH_SOBEL_MAG = (30, 100)
    # Dimension (meter/pixel)
    Y_MPP = 30. / 720
    X_MPP = 3.7 / 700

    def __init__(self, camcal):
        self.camcal = camcal

    def pipeline(self, image, plot=False):
        # Undistort
        image_undist = self.camcal.undistort(image, False)
        # Apply thresholds for binary image
        image_corr = self.correction(image_undist)
        if plot:
            plt.imshow(image_corr, cmap='gray')
            plt.show()
        # Warp
        image_warped = self.camcal.warp(image_corr)
        if plot:
            plt.imshow(image_warped)
            plt.show()
        # Find lanes in warped image
        lane_warp, fit_l, fit_r, off = self.find_line(image_warped)
        if plot:
            plt.imshow(lane_warp)
            plt.show()
        # Measure curvature
        r_left, r_right = self.measuring_curvature(lane_warp, fit_l, fit_r)
        # Unwarp the lines
        lane = self.camcal.unwarp(lane_warp)
        # Generate output
        final = self.draw_line(image_undist, lane)
        final = self.background_for_text(final, 100)
        final = self.text_on_image(final, r_left, r_right, off)
        if plot:
            plt.imshow(final)
            plt.show()
        return final

    def correction(self, image):
        # Path 1: Color Mask: Red-Channel and Saturation-Channel
        corr_red = self.color_RGB(image, 0)
        binary_red = self.create_binary(corr_red)
        binary_red[(corr_red >= self.THRESH_CORR_RED[0]) & (corr_red <= self.THRESH_CORR_RED[1])] = 1
        corr_sat = self.color_HLS(image, 2)
        binary_sat = self.create_binary(corr_sat)
        binary_sat[(corr_sat >= self.THRESH_CORR_SAT[0]) & (corr_sat <= self.THRESH_CORR_SAT[1])] = 1
        color_binary = self.create_binary(binary_red)
        color_binary[(binary_red == 1) & (binary_sat == 1)] = 1

        # Path 2: Sobel-Magnitude for Gray-Channel
        sobelx = self.sobel(self.grayscale(image), 'x', self.KERNEL_SOBEL)
        sobely = self.sobel(self.grayscale(image), 'y', self.KERNEL_SOBEL)
        sobel_mag = self.magnitude(sobelx, sobely)
        sobel_binary = self.create_binary(sobel_mag)
        sobel_binary[(sobel_mag >= self.THRESH_SOBEL_MAG[0]) & (sobel_mag <= self.THRESH_SOBEL_MAG[1])] = 1

        # Combine paths
        combined_binary = self.create_binary(binary_red)
        combined_binary[(color_binary == 1) | (sobel_binary == 1)] = 1
        # Mask region of interest
        masked = self.region_of_interest(combined_binary, self.VERTICES)
        return masked

    def find_line(self, binary_image, nwindows=9, margin=100, min_pixels=10, plot=False):
        # Take a histogram of the bottom half of the image
        histogram = np.sum(binary_image[binary_image.shape[0] // 2:, :], axis=0)
        # Create an output image to draw on and visualize the result
        out_img = np.dstack((binary_image, binary_image, binary_image)) * 255
        # Identify the peaks in left and right half
        midpoint = np.int(histogram.shape[0] / 2)
        left_x_base = np.argmax(histogram[:midpoint])
        right_x_base = np.argmax(histogram[midpoint:]) + midpoint
        # Find window height
        window_height = binary_image.shape[0] // nwindows
        # Find indices where binary is not zero
        nonzero = binary_image.nonzero()
        nonzero_y = np.array(nonzero[0])
        nonzero_x = np.array(nonzero[1])
        # Current positions to be updated for each window
        left_x_current = left_x_base
        right_x_current = right_x_base
        # Create empty lists to receive left and right lane pixel indices
        left_lane_inds = []
        right_lane_inds = []
        # Step through the windows one by one
        for window in range(nwindows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = binary_image.shape[0] - (window + 1) * window_height
            win_y_high = binary_image.shape[0] - window * window_height
            win_x_left_low = left_x_current - margin
            win_x_left_high = left_x_current + margin
            win_x_right_low = right_x_current - margin
            win_x_right_high = right_x_current + margin
            # Draw the windows on the visualization image
            if plot == True:
                cv2.rectangle(out_img, (win_x_left_low, win_y_low), (win_x_left_high, win_y_high), (0, 255, 0), 2)
                cv2.rectangle(out_img, (win_x_right_low, win_y_low), (win_x_right_high, win_y_high), (0, 255, 0), 2)
                # Identify the nonzero pixels in x and y within the window
            good_left_inds = ((nonzero_y >= win_y_low) &
                              (nonzero_y < win_y_high) &
                              (nonzero_x >= win_x_left_low) &
                              (nonzero_x < win_x_left_high)).nonzero()[0]
            good_right_inds = ((nonzero_y >= win_y_low) &
                               (nonzero_y < win_y_high) &
                               (nonzero_x >= win_x_right_low) &
                               (nonzero_x < win_x_right_high)).nonzero()[0]
            # Append these indices to the lists
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)
            # If you found > minpix pixels, recenter next window on their mean position
            if len(good_left_inds) > min_pixels:
                left_x_current = np.int(np.mean(nonzero_x[good_left_inds]))
            if len(good_right_inds) > min_pixels:
                right_x_current = np.int(np.mean(nonzero_x[good_right_inds]))
        # Concatenate the arrays of indices
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
        # Extract left and right line pixel positions
        left_x = nonzero_x[left_lane_inds]
        left_y = nonzero_y[left_lane_inds]
        right_x = nonzero_x[right_lane_inds]
        right_y = nonzero_y[right_lane_inds]
        # Fit a second order polynomial to each
        left_fit = np.polyfit(left_y, left_x, 2)
        right_fit = np.polyfit(right_y, right_x, 2)
        # Build mean over the last 13 frames
        if (np.sign(left_fit[0]) == np.sign(right_fit[0])):
            del self.left_history[0]
            self.left_history.append(list(left_fit))
            del self.right_history[0]
            self.right_history.append(list(right_fit))
            left_fit_nonzero = []
            for item in self.left_history:
                if item != [0, 0, 0]:
                    left_fit_nonzero.append(item)
            right_fit_nonzero = []
            for item in self.right_history:
                if item != [0, 0, 0]:
                    right_fit_nonzero.append(item)
            self.left_mean = np.mean(left_fit_nonzero, axis=0)
            self.right_mean = np.mean(right_fit_nonzero, axis=0)
        # Generate x and y values for plotting
        plot_y = np.linspace(0, binary_image.shape[0] - 1, binary_image.shape[0])
        left_fit_x = self.left_mean[0] * plot_y ** 2 + self.left_mean[1] * plot_y + self.left_mean[2]
        right_fit_x = self.right_mean[0] * plot_y ** 2 + self.right_mean[1] * plot_y + self.right_mean[2]
        out_img[nonzero_y[left_lane_inds], nonzero_x[left_lane_inds]] = [255, 0, 0]
        out_img[nonzero_y[right_lane_inds], nonzero_x[right_lane_inds]] = [0, 0, 255]
        img_lane = np.zeros_like(out_img)
        lane_left = np.array([np.transpose(np.vstack([left_fit_x, plot_y]))])
        lane_right = np.array([np.flipud(np.transpose(np.vstack([right_fit_x, plot_y])))])
        lane = np.hstack((lane_left, lane_right))
        cv2.fillPoly(img_lane, np.int_([lane]), (0, 255, 0))
        # Calculations for offset to lane center
        x_l = np.mean(left_fit_x[-20:-1])
        x_r = np.mean(right_fit_x[-20:-1])
        off = midpoint - 0.5 * (x_r - x_l) - x_l
        if plot == True:
            result = cv2.addWeighted(out_img, 1, img_lane, 0.3, 0)
            plt.imshow(result)
            plt.plot(left_fit_x, plot_y, color='yellow')
            plt.plot(right_fit_x, plot_y, color='yellow')
            plt.xlim(0, 1280)
            plt.ylim(720, 0)
            plt.plot([0.5 * binary_image.shape[1], 0.5 * binary_image.shape[1]],
                     [binary_image.shape[0], binary_image.shape[0] - 100], '--r')
            plt.show()
        fit_left_rw = np.polyfit(left_y * self.Y_MPP, left_x * self.X_MPP, 2)
        fit_right_rw = np.polyfit(right_y * self.Y_MPP, right_x * self.X_MPP, 2)
        off_rw = off * self.X_MPP
        return img_lane, fit_left_rw, fit_right_rw, off_rw

    def measuring_curvature(self, img, fit_left, fit_right):
        eval_point = img.shape[0]
        rad_left = ((1 + (2 * fit_left[0] * eval_point * self.Y_MPP + fit_left[1]) ** 2) ** 1.5) / np.absolute(
            2 * fit_left[0])
        rad_right = ((1 + (2 * fit_right[0] * eval_point * self.Y_MPP + fit_right[1]) ** 2) ** 1.5) / np.absolute(
            2 * fit_right[0])
        return rad_left, rad_right

    @staticmethod
    def draw_line(img, img_lane):
        return cv2.addWeighted(img, 1, img_lane, 0.3, 0)

    @staticmethod
    def background_for_text(img, height):
        img[:height, :, :] = img[:height, :, :] * 0.4
        return img

    @staticmethod
    def text_on_image(img, rad_left, rad_right, off_m):
        font = cv2.FONT_HERSHEY_SIMPLEX
        color = (255,) * 3
        # Radius left lane
        cv2.putText(img, ('Left:'), (20, 40), font, 1.2, color, 2)
        cv2.putText(img, ('%.1f m' % rad_left), (160, 40), font, 1.2, color, 2)
        # Radius right lane
        cv2.putText(img, ('Right:'), (20, 80), font, 1.2, color, 2)
        cv2.putText(img, ('%.1f m' % rad_right), (160, 80), font, 1.2, color, 2)
        # Offset to lane center
        cv2.putText(img, ('Offset:'), (500, 40), font, 1.2, color, 2)
        cv2.putText(img, ('%.1f m' % off_m), (500, 80), font, 1.2, color, 2)
        return img

        # HELPER FUNCTIONS
    @staticmethod
    def grayscale(image):
        img = np.copy(image)
        return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    @staticmethod
    def color_HLS(image, channel='2'):
        img = np.copy(image)
        hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float)
        hls_channel = hls[:, :, channel]
        return hls_channel

    @staticmethod
    def color_RGB(img, channel='0'):
        rgb_channel = np.copy(img[:, :, channel])
        return rgb_channel

    @staticmethod
    def sobel(img, drct='x', ksize=3):
        if drct == 'x':
            sobel = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=ksize)
        if drct == 'y':
            sobel = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=ksize)
        return sobel

    @staticmethod
    def absolute(value):
        return np.absolute(value)

    @staticmethod
    def magnitude(sobelx, sobely):
        abs_mag = np.sqrt(sobelx ** 2 + sobely ** 2)
        scaled_mag = np.uint8(255 * abs_mag / np.max(abs_mag))
        return scaled_mag

    @staticmethod
    def direction(abs_sobely, abs_sobelx):
        return np.arctan2(abs_sobely, abs_sobelx)

    @staticmethod
    def scaled_sobel(abs_sobel):
        return np.uint8(255 * abs_sobel / np.max(abs_sobel))

    @staticmethod
    def create_binary(image):
        return np.zeros_like(image)

    @staticmethod
    def region_of_interest(img, vertices):
        mask = np.zeros_like(img)
        if len(img.shape) > 2:
            channel_count = img.shape[2]
            ignore_mask_color = (255,) * channel_count
        else:
            ignore_mask_color = 255
        cv2.fillPoly(mask, vertices, ignore_mask_color)
        masked_image = cv2.bitwise_and(img, mask)
        return masked_image

run = False

if run:
    camcal = CameraCalibration()
    camcal.calibrate_camera()
    det = LaneLineDetector(camcal)

    output = 'harder_challenge_video_submit.mp4'
    clip1 = VideoFileClip('harder_challenge_video.mp4')
    clip = clip1.fl_image(det.pipeline)
    clip.write_videofile(output, audio=False)