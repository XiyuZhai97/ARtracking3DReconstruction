import CalibrationHelpers as calib
print("calib folder:")
libname = input()
intrinsics, distortion, roi, new_intrinsics, mean_err= calib.CalibrateCamera(libname,True)
if mean_err < 1:
    calib.SaveCalibrationData(libname, intrinsics, distortion, new_intrinsics, roi)