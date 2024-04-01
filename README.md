# various_angles-chessboard-calibration
다양한 각도에서 찍은 체스판 보정

import numpy as np
import cv2 as cv

    def select_img_from_video(video_file, board_pattern, select_all=False, wait_msec=10):
        video = cv.VideoCapture(video_file)  
        img_select = []
        while True:
            ret, frame = video.read()
            if not ret:
                break
            gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            complete, pts = cv.findChessboardCorners(gray, board_pattern)
            if complete:
                img_select.append(frame)
                if not select_all:
                    break
            cv.imshow('Select images from video', frame)
            if cv.waitKey(wait_msec) & 0xFF == ord('q'):
                break
        video.release()
        cv.destroyAllWindows()
        return img_select

    def calib_camera_from_chessboard(images, board_pattern, board_cellsize, K=None, dist_coeff=None, calib_flags=None):
        img_points = []
        for img in images:
            gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            complete, pts = cv.findChessboardCorners(gray, board_pattern)
            if complete:
                img_points.append(pts)
            else:
                print("Warning: Incomplete chessboard points found in one or more images.")
        assert len(img_points) > 0, 'There is no set of complete chessboard points!'
        
        
        obj_pts = np.zeros((board_pattern[0] * board_pattern[1], 3), np.float32)
        obj_pts[:, :2] = np.mgrid[0:board_pattern[0], 0:board_pattern[1]].T.reshape(-1, 2)
        obj_pts *= board_cellsize
        obj_points = [obj_pts] * len(img_points)
    
        
        return cv.calibrateCamera(obj_points, img_points, gray.shape[::-1], K, dist_coeff, flags=calib_flags)

if __name__ == "__main__"
    video_file = 'C:/Users/user/Downloads/chessboard.mp4' 
    board_pattern = (9, 7) 
    board_cellsize = 0.5  
    selected_images = select_img_from_video(video_file, board_pattern)

    try:
        
        ret, K, dist_coeff, rvecs, tvecs = calib_camera_from_chessboard(selected_images, board_pattern, board_cellsize)

       
        print("A number of applied images =", len(selected_images))
        print("RMS error =", ret)
        print("Camera matrix (K) =\n", K)
        print("Distortion coefficient (k1, k2, p1, p2, k3, ...) =\n", dist_coeff)
    except AssertionError:
        print("Unable to calibrate from video, trying with chessboard image.")

       
        board_image = cv.imread('chessboard.jpg')
        board_gray = cv.cvtColor(board_image, cv.COLOR_BGR2GRAY)
        complete, pts = cv.findChessboardCorners(board_gray, board_pattern, flags=cv.CALIB_CB_ADAPTIVE_THRESH + cv.CALIB_CB_NORMALIZE_IMAGE)
        assert complete, 'Unable to find chessboard corners in the image!'

       
        ret, K, dist_coeff, rvecs, tvecs = calib_camera_from_chessboard([board_image], board_pattern, board_cellsize)

      
        print("RMS error =", ret)
        print("Camera matrix (K) =\n", K)
        print("Distortion coefficient (k1, k2, p1, p2, k3, ...) =\n", dist_coeff)

위코드는 동영상으로 찍은 체스판에서 체스판을 인식하여
Calibration Results를 얻어 distortion correction을 하는 과정이지만
[chessboard.mp4]
![image](https://github.com/kohjun/various_angles-chessboard-calibration/assets/82298792/98924c87-a68c-4770-a481-7fcece2377ff)
[chessboard.jpg]

![image](https://github.com/kohjun/various_angles-chessboard-calibration/assets/82298792/8998dcbf-ffe2-425d-9743-75c80e8b9022)

**[코드를 실행한 결과]**
Exception has occurred: AssertionError
Unable to find chessboard corners in the image!
File "C:\Users\user\Downloads\chessboard.py", line 60, in <module>
ret, K, dist_coeff, rvecs, tvecs = calib_camera_from_chessboard(selected_images, board_pattern, board_cellsize)
                                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
File "C:\Users\user\Downloads\chessboard.py", line 36, in calib_camera_from_chessboard
assert len(img_points) > 0, 'There is no set of complete chessboard points!'
     ^^^^^^^^^^^^^^^^^^^
AssertionError: There is no set of complete chessboard points!
During handling of the above exception, another exception occurred:


위와 같은 오류 발생으로 동영상과 인터넷 상에서 구한 이미지에서 체스판을 인식하지 못하였다. 
해상도와 이미지의 품질 또는 체스판의 패턴과 셀의 크기를 인식하지 못한 결과인 것 같다.
