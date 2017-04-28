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
    printf("Camera1: %f %f %f %f %f %f %d %d\n", camera1.K.cx , camera1.K.cy ,camera1.K.fx ,camera1.K.fy ,
            camera1.K.k1 , camera1.K.k2, camera1.K.width ,camera1.K.height ) ;   

    cv::Mat Frame;
    if (!camera1.openCamera(2)) {
        printf("Open camera failed!\n");
        exit(0);
    }

    ImageFrame *refImgFrame = NULL;

    char cmd = ' ';
    while (true) {

        if ( !camera1.getFrame(Frame, camera1.BGR) ) {
            printf("Get Frame failed!\n");
            break;
        }

        if (cmd == 's') {
            refImgFrame = new ImageFrame(Frame, &K1); 
            refImgFrame->extractFAST();
        } else {
            if ( refImgFrame != NULL ) {
                ImageFrame newImgFrame(Frame, &K1);
                
                // optical flow result
                newImgFrame.opticalFlowFAST(*refImgFrame);
                for (int i = 0, _end = (int)newImgFrame.trackedPoints.size(); i < _end; i++) { // draw result
                    if (newImgFrame.trackedPoints[i].x >= 0) {
                        cv::line(Frame, refImgFrame->points[i], 
                                newImgFrame.trackedPoints[i],
                                cv::Scalar(0, 255, 0));
                    }
                }

                // SBI alignment
                //newImgFrame.SBITrackFAST(*refImgFrame);
                //for (int i = 0, _end = (int)newImgFrame.keyPoints.size(); i < _end; i++) { // draw result
                //    if (newImgFrame.keyPoints[i].pt.x >= 0) {
                //        cv::line(Frame, refImgFrame->keyPoints[i].pt, 
                //                newImgFrame.keyPoints[i].pt,
                //                cv::Scalar(255, 0, 0));
                //    }
                //}
            }
        }

        

        cv::imshow("result",Frame);

        cmd = cv::waitKey(33);
    }
    

    return 0;
}
