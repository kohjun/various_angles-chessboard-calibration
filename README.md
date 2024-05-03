# various_angles-chessboard-calibration
다양한 각도에서 찍은 체스판 보정

import numpy as np
import cv2 as cv

def calib_camera_from_chessboard(image, board_pattern, board_cellsize, K=None, dist_coeff=None, calib_flags=None):
    # Convert the image to grayscale
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    
    # Find 2D corner points from the image
    complete, pts = cv.findChessboardCorners(gray, board_pattern)
    if not complete:
        raise ValueError('Complete chessboard points not found in the image!')
    
    # Prepare 3D points of the chessboard
    obj_pts = np.zeros((board_pattern[0] * board_pattern[1], 3), np.float32)
    obj_pts[:, :2] = np.mgrid[0:board_pattern[0], 0:board_pattern[1]].T.reshape(-1, 2) * board_cellsize
    
    # Calibrate the camera
    return cv.calibrateCamera([obj_pts], [pts], gray.shape[::-1], K, dist_coeff, flags=calib_flags)

if __name__ == '__main__':
    # Example usage with a single image
    image_file = 'chessboard.jpg'
    board_pattern = (9, 6)  # Chessboard pattern size
    board_cellsize = 0.025  # Size of each square on the chessboard in meters
    
    # Load the image
    image = cv.imread(image_file)
    
    # Perform camera calibration
    ret, K, dist_coeff, _, _ = calib_camera_from_chessboard(image, board_pattern, board_cellsize)
    
    # Print camera matrix and distortion coefficients
    print("Camera matrix:")
    print(K)
    print("Distortion coefficients:")
    print(dist_coeff)



import numpy as np
import cv2 as cv

def distortion_correction(image, K, dist_coeff):
    # Convert distortion coefficients to 1D array
    dist_coeff = dist_coeff[0]

    # Undistort the image
    undistorted_image = cv.undistort(image, K, dist_coeff)
    
    # Display the original and undistorted images
    cv.imshow('Original Image', image)
    cv.imshow('Undistorted Image', undistorted_image)
    cv.waitKey(0)
    cv.destroyAllWindows()

if __name__ == '__main__':
    # Example usage with a single image
    image_file = 'chessboard.jpg'
    K = np.array([[478.13958117, 0, 294.24822921],
                  [0, 480.88817142, 225.95312616],
                  [0, 0, 1]])  # Camera matrix
    dist_coeff = np.array([0.423655594, -2.92047309, -0.00104729862, 0.00371282001, 5.61525755])  # Distortion coefficients
    
    # Load the image
    image = cv.imread(image_file)
    
    # Perform distortion correction
    distortion_correction(image, K, dist_coeff)


위코드는 동영상으로 찍은 체스판에서 체스판을 인식하여
Calibration Results를 얻어 distortion correction을 하는 과정


변환 전 

![image](https://github.com/kohjun/various_angles-chessboard-calibration/assets/82298792/32780999-3685-4426-a569-c950b9d854c8)


변환 후


![image](https://github.com/kohjun/various_angles-chessboard-calibration/assets/82298792/baba5eba-79ee-47a8-936f-319ff2b931cc)
