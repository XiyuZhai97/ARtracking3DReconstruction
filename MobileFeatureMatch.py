import cv2
import numpy as np
import CalibrationHelpers as calib
import glob
import time
import scipy.io as spio
relR = spio.loadmat("relR.mat")['R']
relT = spio.loadmat("relT.mat")['T']

# a function that takes in: the camera intrinsic matrix, 
# a set of matches between two images, a set of points in each image, 
# and a rotation and translation between the images, and a threshold parameter.
# The function should return an array of either 0 or 1 for each point, 
# where 1 represents an inlier and 0 an outlier (outlier = incorrect match). 
filterflag = False
def FilterByEpipolarConstraint(intrinsics, matches, points1, points2, Rx1, Tx1,
                               threshold = 0.01):
    # your code here
    inlier_mask = 0
    return inlier_mask 

images = glob.glob('Mobile_Ref_data'+'/*.jpeg')
# Load the reference image that we will try to detect in the webcam
reference = cv2.imread(images[0],0)
RES = 480
reference = cv2.resize(reference,(RES,RES))
# create the feature detector. This will be used to find and describe locations
# in the image that we can reliably detect in multiple images
feature_detector = cv2.BRISK_create(octaves=5)
# compute the features in the reference image
reference_keypoints, reference_descriptors = \
        feature_detector.detectAndCompute(reference, None)

keypoint_visualization = cv2.drawKeypoints(
        reference,reference_keypoints,outImage=np.array([]), 
        flags = cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# display the image
cv2.imshow("Keypoints",keypoint_visualization)
matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
# Load the camera calibration matrix
intrinsics, distortion, new_intrinsics, roi = \
        calib.LoadCalibrationData('mobile_calib_data')

imgNum = 0
for fname in images:
    # read the image
    print(fname)
    cap = cv2.imread(fname)
# while True:
    # read the current frame from the webcam
    current_frame = cv2.resize(cap,(RES,RES))
    
    # undistort the current frame using the loaded calibration
    current_frame = cv2.undistort(current_frame, intrinsics, distortion, None,\
                                  new_intrinsics)
    # apply region of interest cropping
    x, y, w, h = roi
    current_frame = current_frame[y:y+h, x:x+w]
    
    # detect features in the current image
    current_keypoints, current_descriptors = \
        feature_detector.detectAndCompute(current_frame, None)
        
    # match the features from the reference image to the current image

    matches = matcher.match(reference_descriptors, current_descriptors)
    print("matches", matches)
    if(filterflag):
        inlier_mask = FilterByEpipolarConstraint(intrinsics, matches, points1, points2, Rx1, Tx1, threshold = 0.01)
                               
        match_visualization = cv2.drawMatches(reference, reference_keypoints, current_frame,
                            current_keypoints, matches, 0, 
                            matchesMask =inlier_mask, #this applies your inlier filter
                            flags=
                            cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
    else:
        match_visualization = cv2.drawMatches(reference, reference_keypoints, current_frame,
                                current_keypoints, matches, 0, 
                                flags=
                                cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)

    cv2.imshow('matches',match_visualization)
    k = cv2.waitKey(1)
    time.sleep(2)
    if k == 27 or k==113:  #27, 113 are ascii for escape and q respectively
        #exit
        break