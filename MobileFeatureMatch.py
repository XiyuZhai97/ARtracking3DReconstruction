import cv2
import numpy as np
import CalibrationHelpers as calib
import glob
import time
import scipy.io as spio
import open3d as o3d
relR = spio.loadmat("relR.mat")['R']
relT = spio.loadmat("relT.mat")['T']

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
    inlier_mask = np.zeros(points1.shape[0])
    real_inlier_mask = np.zeros(points1.shape[0])
    for i in range(points1.shape[0]):
        points1_3[i] = [(points1[i][0] - cx)/fx, (points1[i][1] - cy)/fy, 1]
        points2_3[i] = [(points2[i][0] - cx)/fx, (points2[i][1] - cy)/fy, 1]
        real_inlier_mask[i] = abs(points2_3[i].dot(E).dot(points1_3[i].T))
        if abs(points2_3[i].dot(E).dot(points1_3[i].T)) < threshold:
            inlier_mask[i] = 1
        else:inlier_mask[i] = 0
    return inlier_mask 

images = glob.glob('Mobile_Ref_data'+'/*.jpeg')
images.sort()
reference = cv2.imread(images[0])
RES = 480
reference = cv2.resize(reference,(RES,RES))
feature_detector = cv2.BRISK_create(octaves=5)
reference_keypoints, reference_descriptors = \
        feature_detector.detectAndCompute(reference, None)
keypoint_visualization = cv2.drawKeypoints(
        reference,reference_keypoints,outImage=np.array([]), 
        flags = cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv2.imshow("Keypoints",keypoint_visualization)
matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
intrinsics, distortion, new_intrinsics, roi = \
        calib.LoadCalibrationData('mobile_calib_data')
fx = intrinsics[0][0]
fy = intrinsics[1][1]
cx = intrinsics[0][2]
cy = intrinsics[1][2]
imgNum = 0
feature_track = {}
for fname in images:
    print(fname)
    cap = cv2.imread(fname)
    current_frame = cv2.resize(cap,(RES,RES))
    current_frame = cv2.undistort(current_frame, intrinsics, distortion, None,\
                                  new_intrinsics)
    x, y, w, h = roi
    current_frame = current_frame[y:y+h, x:x+w]
    current_keypoints, current_descriptors = \
        feature_detector.detectAndCompute(current_frame, None)    
    matches = matcher.match(reference_descriptors, current_descriptors)
    for m in matches:
        # print(m.queryIdx)
        if m.queryIdx in feature_track:
            feature_track[m.queryIdx] += 1 
        else:
            feature_track[m.queryIdx] = 1 
    imgNum += 1
cv2.destroyAllWindows()
print("feature_track", len(feature_track))
imgNum = 1
M = np.zeros((0, len(reference_keypoints) + 1))

for fname in images[1:]:
    
    print(fname)
    print("relR", relR[imgNum])
    print("relT", relT[imgNum])

    cap = cv2.imread(fname)
    current_frame = cv2.resize(cap,(RES,RES))
    current_frame = cv2.undistort(current_frame, intrinsics, distortion, None,\
                                  new_intrinsics)
    x, y, w, h = roi
    current_frame = current_frame[y:y+h, x:x+w]
    current_keypoints, current_descriptors = \
        feature_detector.detectAndCompute(current_frame, None)
    matches = matcher.match(reference_descriptors, current_descriptors)
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
    # delet feature_track < 4
    while i < len(matches):
        if feature_track[matches[i].queryIdx] < 4:
            del matches[i]
            i -= 1
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

    E = np.cross(relT[imgNum], relR[imgNum], axisa = 0, axisb = 0)
    threshold = 0.01
    curpoints_3 = np.zeros((imagePoints.shape[0], 3))
    refpoints_3 = np.zeros((referencePoints.shape[0], 3))
    for i in range(imagePoints.shape[0]):
        curpoints_3[i] = [(imagePoints[i][0] - cx)/fx, (imagePoints[i][1] - cy)/fy, 1]
        refpoints_3[i] = [(referencePoints[i][0] - cx)/fx, (referencePoints[i][1] - cy)/fy, 1]
        if abs(np.matmul(np.matmul(refpoints_3[i], E), np.transpose(curpoints_3[i]))) > threshold:
            curpoints_3[i] = [0, 0, 0]
            refpoints_3[i] = [0, 0, 0]
    k = 0
    print("curpoints_3.shape", curpoints_3.shape)

    while k < curpoints_3.shape[0]:
        if np.sum(curpoints_3[k, :]) == 0:
            curpoints_3 = np.delete(curpoints_3, k, axis = 0)
            refpoints_3 = np.delete(refpoints_3, k, axis = 0)
            k -= 1
        k += 1
    print("curpoints_3.shape", curpoints_3.shape)
    # i = 0
    # if abs(refpoints_3[i].dot(E).dot(curpoints_3[i].T)) > threshold:
    #     curpoints_3 = np.delete(curpoints_3, i, axis = 0)
    #     points_3 = np.delete(refpoints_3, i, axis = 0)
    Mtemp = np.zeros((3*len(matches), len(reference_keypoints) + 1))
    for i in range(curpoints_3.shape[0]):
        Mtemp[i*3: i*3+3, matches[i].queryIdx] = np.cross(curpoints_3[i], np.matmul(relR[imgNum],refpoints_3[i]), axisa = 0, axisb = 0)
        Mtemp[i*3:i*3+3, -1] = np.cross(curpoints_3[i], relT[imgNum], axisa = 0, axisb = 0)
    print("Mtemp", Mtemp.shape)

    M = np.append(M, Mtemp, axis=0)
    imgNum += 1
    k = cv2.waitKey(1)
    time.sleep(2)
    if k == 27 or k==113:  #27, 113 are ascii for escape and q respectively
        #exit
        break

referencePoints = np.float32([rp.pt \
                            for rp in reference_keypoints])

refpoints_3 = np.zeros((referencePoints.shape[0], 3))
for i in range(referencePoints.shape[0]):
    refpoints_3[i] = [(referencePoints[i][0] - cx)/fx, (referencePoints[i][1] - cy)/fy, 1]

print("M.shape before",M.shape)    
print("referencePoints", refpoints_3.shape)
k = 0
while k < M.shape[1]:
    if np.sum(M[:, k]) == 0:
        M = np.delete(M, k, axis = 1)
        refpoints_3 = np.delete(refpoints_3, k, axis = 0)
        k -= 1
    k += 1
k = 0
while k < M.shape[0]:
    if np.sum(M[k, :]) == 0:
        M = np.delete(M, k, axis = 0)
        k -= 1
    k += 1
print("M.shape", M.shape)
print("referencePoints", refpoints_3.shape)

spio.savemat("M.mat", {"M":M}) # .reshape((relT.size//3, 3))})
print("M", M)
your_pointCloud = refpoints_3
W,U,Vt = cv2.SVDecomp(M)

depths = Vt[-1,:]/Vt[-1,-1]

for i in range(len(refpoints_3)):
    if abs(depths[i]) < 3:
        your_pointCloud[i,2] *= depths[i]
        # your_pointCloud[i] = refpoints_3[i] * [depths[i], depths[i], depths[i]]

# print(np.max(your_pointCloud, axis = 0))
# print(np.where(your_pointCloud == np.max(your_pointCloud, axis = 0)[2]))
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(your_pointCloud)
o3d.visualization.draw_geometries([pcd])
