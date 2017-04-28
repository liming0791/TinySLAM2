#include <stdio.h>
#include <stdlib.h>

#include <iostream>

#include <opencv2/opencv.hpp>

#include "CameraDevice.h"
#include "ImageFrame.h"

using namespace std;

int main(int argc, char** argv)
{

    CameraIntrinsic K1(argv[1]);
    CameraDevice camera1(K1);
    printf("Camera1: %f %f %f %f %f %d %d\n", camera1.K.cx , camera1.K.cy ,
            camera1.K.fx ,camera1.K.fy ,camera1.K.k1 ,camera1.K.width ,camera1.K.height ) ;   

    cv::Mat Frame;
    if (!camera1.openCamera(1)) {
        printf("Open camera failed!\n");
        exit(0);
    }

    while (true) {

        if ( !camera1.getFrame(Frame, camera1.BGR) ) {
            printf("Get Frame failed!\n");
            break;
        }

        cv::Mat UndistortFrame(480, 640, CV_8UC3);
        cv::Mat UndistortFrame2(480, 640, CV_8UC3);
        for (int r = 0; r < Frame.rows; r++) {
            for (int c = 0; c < Frame.cols; c++) {
                cv::Point2f np = camera1.K.undistort(c, r); 
                if (np.y >= 0 && np.y < Frame.rows && np.x >=0 && np.x < Frame.cols) {
                    UndistortFrame2.at<cv::Vec3b>(np.y, np.x) =
                            Frame.at<cv::Vec3b>(r, c);
                }
                cv::Point2f op = camera1.K.distort(c, r);
                if (op.y >= 0 && op.y < Frame.rows && op.x >=0 && op.x < Frame.cols) {
                    UndistortFrame.at<cv::Vec3b>(r, c) =
                            Frame.at<cv::Vec3b>(op.y, op.x);
                }
            }
        }

        cv::imshow("origin",Frame);
        cv::imshow("undistort", UndistortFrame);
        cv::imshow("undistort2", UndistortFrame2);

        cv::waitKey(33);
    }
    

    return 0;
}
