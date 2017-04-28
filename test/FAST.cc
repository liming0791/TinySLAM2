#include <stdio.h>
#include <stdlib.h>

#include <iostream>
#include <thread>

#include <opencv2/opencv.hpp>

#include "CameraDevice.h"
#include "ImageFrame.h"
#include "Timer.h"

using namespace std;

int main(int argc, char** argv)
{

    // camera device 
    CameraIntrinsic K1(argv[1]);
    CameraDevice camera1(K1);
    printf("Camera1: %f %f %f %f %f %f %d %d\n", camera1.K.cx , camera1.K.cy ,camera1.K.fx ,camera1.K.fy ,
            camera1.K.k1 , camera1.K.k2, camera1.K.width ,camera1.K.height ) ;   

    bool isDataset = false;
    bool isVideo = false;
    cv::Mat Frame;
    if (!camera1.openDataset(argv[2])) {
        printf("Open dataset failed!\nTry open video file\n");

        if (!camera1.openVideo(argv[2])) {
            printf("Open video failed!\nTry open camera\n");

            if (!camera1.openCamera(atoi(argv[2]))) {
                printf("Open camera failed!\n");
                exit(0);
            }
        } else {
            isVideo = true;
        }
    } else {
        isDataset = true;
    }
    
    char cmd = ' ';
    cv::namedWindow("result");

    while (true) {

        if ( !camera1.getFrame(Frame, camera1.BGR) ) {
            printf("Get Frame failed!\n");
            break;
        }

        // If is dataset, control at each image
        if (isDataset || isVideo) {
            cmd = cv::waitKey(-1);
        }

        ImageFrame imageFrame;

        TIME_BEGIN();
        imageFrame = ImageFrame(Frame, &K1); 
        TIME_END("Construct a imageFrame");

        TIME_BEGIN();
        imageFrame.extractFAST();
        TIME_END("extract FAST");

        for (int i = 0, _end = (int)imageFrame.measure2ds.size(); i < _end; i++) {
            Measure2D* pmeasure2d = &imageFrame.measure2ds[i];
            cv::Scalar color;
            switch (pmeasure2d->levelIdx) {
                case 3: color = cv::Scalar(0,0,170); break;
                case 2: color = cv::Scalar(30, 255, 255); break;
                case 1: color = cv::Scalar(255, 207, 0); break;
                case 0: color = cv::Scalar(246, 1, 0); break;
            }
            cv::circle(Frame, pmeasure2d->pt, pmeasure2d->levelIdx*2+3, color);
        }

        cv::imshow("result",Frame);
        cmd = cv::waitKey(33);
    }

    return 0;
}
