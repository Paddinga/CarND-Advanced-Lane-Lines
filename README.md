
# Self-Driving Car Engineer Nanodegree


## Project 4: **Advanced Lane Finding** 

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

***

[//]: # (Image References)

[image1]: ./md_images/chessboard_undist.png "Undistorted chessboard"
[image2]: ./md_images/road_undist.png "Undistorted road"
[image3]: ./md_images/warp_source.png "Source image for warp"
[image4]: ./md_images/warp_destination.png "Destination image for warp"
[image5]: ./md_images/image_correction.png "Image correction"
[image6]: ./md_images/warped.png "Warped binary"
[image7]: ./md_images/lines_found.png "Lines detected"
[image8]: ./md_images/final.png "Final image"


### Camera calibration and undistort images

The calibration of the camera is processed in the method calibrate_camera() in the class CameraCalibration of camera_calibration.py by extracting the calibration parameter M from the 20 provided calibration images and generate the undistortion parameters. With transformation is then processed with the method undistort().

![alt text][image1]

The undistortion is then applied to the road images.

![alt text][image2]

### Warp image

Before applying any image correction I prepared the warp parameters. I used the straight_lines2.jpg and marked four points suitable for the transformation with an "x". This was an iterative process until I was happy with the positioning of the markings. 

![alt text][image3]

By using the method warp() in class CameraCalibration I transformed the images. The destination points were chosen by rule-of-thumb to have the resulting lines nicely centered.

![alt text][image4]

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 602, 445      | 340, 0        | 
| 680, 445      | 940, 0        |
| 1040, 675     | 940, 720      |
| 275, 675      | 340, 720      |

### Lane line detection pipeline

The class LaneLineDetector in detection_pipeline.py involves the necessary methods for image praparation, lane detection and generating the output image for the detected lines and related values for angles and offset.

#### Image Correction

For image preparation I use three parallel paths combined in one output image.

1. Red Channel from RGB with Threshold
2. Saturation Channel from HLS with Threshold
3. Sobel-magnitude on Graychannel

The color channels are combined with an AND. This suits to detect white lines as well as yellow lines. In case this doesn't detect any lines, the Sobel-magnitude detects several edges around the lines to cover. The color and sobel detection is combined with OR. 
Though it seems that the Sobel does bring to much noise into the binary images, it seems that a bit of noise is very helpful in case the color-based detection doesn't work.
The thresholds are defined by testing each of the paths with test images.

To delete detections in the background and on other lanes a region of interest is applied to the image.

![alt text][image5]

#### Warping

After preprocessing the images are being transformed to birdseye-view.

![alt text][image6]

#### Find lines

The lane detection is pretty similar to the code in the courses with the sliding window.

![alt text][image7]

To smoothen the output I added two checks starting in #150 of detection_pipeline.py: 

1. Are the signs of both of the lines, left and right, likewise.
2. There is a history array for each left and right lines with a size of 13. From this history the mean of the lines is processed and handed to calculate the polyfit.

#### Unwarp and calculate angles and offset

The next step is to calculate the angles for left and right line as well as the offset to the center of the lane.
These calculations are processed as shown in the course. Afterwards the images of the detected lanes are being unwarped, combined with the original (undistorted) image and the calculated values are added to the image.

![alt text][image8]

### Pipeline for video

The pipeline for the video can be activated by setting run = True in the detection_pipeline.py
Therefore the input video and the name for the output needs to be chosen.

```python
run = False

if run:
    camcal = CameraCalibration()
    camcal.calibrate_camera()
    det = LaneLineDetector(camcal)

    output = 'output.mp4'
    clip1 = VideoFileClip('input_video.mp4')
    clip = clip1.fl_image(det.pipeline)
    clip.write_videofile(output, audio=False)
```

The output for the project video is saved under --> project_video_submit.mp4

I also ran the pipeline for the challenge and the harder-challenge videos. --> challenge_video_submit.mp4 and harder_challenge_video_submit.mp4

### Discussion

This was really an interesting project. My biggest takeaway is the handling of classes.

I am quite happy with the result of the outcome. The output is smooth and really acurate. This is achieved by using the lane history and calcuting the mean over the last 13 frames. On the other hand is this to slow and unflexible to handle the harder_challenge_video. There are also possible improvements with the image correction. And here is the downside of the project. It took quite long to tune all the parameters for the treshholds and this could be done even longer.

In the end it was really fun to work on this project.


