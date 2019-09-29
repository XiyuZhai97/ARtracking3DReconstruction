import cv2
import numpy as np
import CalibrationHelpers as calib
import glob
import time
import scipy.io as spio
import open3d as o3d

relR = spio.loadmat("relR.mat")['R']
relT = spio.loadmat("relT.mat")['T']

# a function that takes in: the camera intrinsic matrix, 
# a set of matches between two images, a set of points in each image, 
# and a rotation and translation between the images, and a threshold parameter.
# The function should return an array of either 0 or 1 for each point, 
# where 1 represents an inlier and 0 an outlier (outlier = incorrect match). 
filterflag = True
def FilterByEpipolarConstraint(intrinsics, matches, points1, points2, Rx1, Tx1,
                               threshold = 0.005):
    # your code here
    fx = intrinsics[0][0]
    fy = intrinsics[1][1]
    cx = intrinsics[0][2]
    cy = intrinsics[1][2]
    points1_3 = np.zeros((points1.shape[0], 3))
    points2_3 = np.zeros((points2.shape[0], 3))
    
    E = np.cross(Tx1, Rx1, axisa = 0, axisb = 0)
    # print(E)
    # print(points1)
    inlier_mask = np.zeros(points1.shape[0])
    real_inlier_mask = np.zeros(points1.shape[0])
    for i in range(points1.shape[0]):
        points1_3[i] = [(points1[i][0] - cx)/fx, (points1[i][1] - cy)/fy, 1]
        points2_3[i] = [(points2[i][0] - cx)/fx, (points2[i][1] - cy)/fy, 1]
        
        real_inlier_mask[i] = abs(points2_3[i].dot(E).dot(points1_3[i].T))
        if abs(points2_3[i].dot(E).dot(points1_3[i].T)) < threshold:
            inlier_mask[i] = 1
        else:inlier_mask[i] = 0
    # print("real", real_inlier_mask)
    # print(inlier_mask)
    return inlier_mask 

images = glob.glob('Mobile_Ref_data'+'/*.jpeg')
images.sort()
# Load the reference image that we will try to detect in the webcam
reference = cv2.imread(images[0])
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
feature_track = np.zeros(len(reference_keypoints)) 
for fname in images:
    # read the image
    print(fname)
    cap = cv2.imread(fname)
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
    for m in matches:
        feature_track[m.queryIdx] += 1 

    # referencePoints = np.float32([reference_keypoints[m.queryIdx].pt \
    #                               for m in matches])
    # imagePoints = np.float32([current_keypoints[m.trainIdx].pt \
    #                               for m in matches])
    # inlier_mask = FilterByEpipolarConstraint(intrinsics, matches, referencePoints, imagePoints, relR[imgNum], relT[imgNum], threshold = 0.01)                 
    # match_visualization = cv2.drawMatches(reference, reference_keypoints, current_frame,
    #                     current_keypoints, matches, 0, 
    #                     matchesMask =inlier_mask, #this applies your inlier filter
    #                     flags=
    #                     cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
    # cv2.imshow('matches_after filter',match_visualization)
    # match_visualization = cv2.drawMatches(reference, reference_keypoints, current_frame,
    #                         current_keypoints, matches, 0, 
    #                         flags=
    #                         cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)

    # cv2.imshow('matches',match_visualization)
    # k = cv2.waitKey(1)
    # # time.sleep(2)
    # if k == 27 or k==113:  #27, 113 are ascii for escape and q respectively
    #     #exit
    #     break
    imgNum += 1
print(feature_track)
cv2.destroyAllWindows()
imgNum = 0

for fname in images[1:]:
    # read the image
    print(fname)
    cap = cv2.imread(fname)
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
    reference_keypoints, reference_descriptors = \
        feature_detector.detectAndCompute(reference, None)
    matches = matcher.match(reference_descriptors, current_descriptors)
    # -Previous
    referencePoints = np.float32([reference_keypoints[m.queryIdx].pt \
                                  for m in matches])
    imagePoints = np.float32([current_keypoints[m.trainIdx].pt \
                                  for m in matches])
    inlier_mask = FilterByEpipolarConstraint(intrinsics, matches, referencePoints, imagePoints, relR[imgNum], relT[imgNum], threshold = 0.01)                 
    match_visualization = cv2.drawMatches(reference, reference_keypoints, current_frame,
                        current_keypoints, matches, 0, 
                        matchesMask =inlier_mask, #this applies your inlier filter
                        flags=
                        cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
    cv2.imshow('matches_after filter',match_visualization)
    match_visualization = cv2.drawMatches(reference, reference_keypoints, current_frame,
                            current_keypoints, matches, 0, 
                            flags=
                            cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)

    cv2.imshow('matches',match_visualization)

    i = 0
    while i < len(matches):
        if feature_track[matches[i].queryIdx] < 3:
            del matches[i]
        i += 1
    referencePoints = np.float32([reference_keypoints[m.queryIdx].pt \
                                  for m in matches])
    imagePoints = np.float32([current_keypoints[m.trainIdx].pt \
                                  for m in matches])
    inlier_mask = FilterByEpipolarConstraint(intrinsics, matches, referencePoints, imagePoints, relR[imgNum], relT[imgNum], threshold = 0.01)  
    match_visualization = cv2.drawMatches(reference, reference_keypoints, current_frame,
                        current_keypoints, matches, 0, 
                        matchesMask =inlier_mask, #this applies your inlier filter
                        flags=
                        cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
    cv2.imshow('track>=4, matches_after filter',match_visualization)
    M = np.zeros((3*len(matches), len(matches) + 1))
    print(len(matches))
    
    fx = intrinsics[0][0]
    fy = intrinsics[1][1]
    cx = intrinsics[0][2]
    cy = intrinsics[1][2]
    points1_3 = np.zeros((imagePoints.shape[0], 3))
    points2_3 = np.zeros((referencePoints.shape[0], 3))

    your_pointCloud = np.zeros((len(matches), 3))

    for i in range(imagePoints.shape[0]):
        points1_3[i] = [(imagePoints[i][0] - cx)/fx, (imagePoints[i][1] - cy)/fy, 1]
        points2_3[i] = [(referencePoints[i][0] - cx)/fx, (referencePoints[i][1] - cy)/fy, 1]

    for i in range(len(matches)):

        M[i*3][i] = np.cross(points1_3[i], relR[imgNum], axisa = 0, axisb = 0).dot(points2_3[i])[0]
        M[i*3+1][i] = np.cross(points1_3[i], relR[imgNum], axisa = 0, axisb = 0).dot(points2_3[i])[1]
        M[i*3+2][i] = np.cross(points1_3[i], relR[imgNum], axisa = 0, axisb = 0).dot(points2_3[i])[2]

        M[i*3][len(matches)] = np.cross(points1_3[i], relT[imgNum], axisa = 0, axisb = 0)[0]
        M[i*3+1][len(matches)] = np.cross(points1_3[i], relT[imgNum], axisa = 0, axisb = 0)[1]
        M[i*3+1][len(matches)] = np.cross(points1_3[i], relT[imgNum], axisa = 0, axisb = 0)[2]
    W,U,Vt = cv2.SVDecomp(M)  
    depths = Vt[-1,:]/Vt[-1,-1]
    print(points2_3)
    for i in range(len(matches)):
        your_pointCloud[i] = points2_3[i] * [1, 1, depths[i]]
    print(your_pointCloud)
    imgNum += 1
    k = cv2.waitKey(1)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(your_pointCloud)
    o3d.visualization.draw_geometries([pcd])

    if k == 27 or k==113:  #27, 113 are ascii for escape and q respectively
        #exit
        break
    elif k == 32: #32 is ascii for space
        continue
    time.sleep(1)
