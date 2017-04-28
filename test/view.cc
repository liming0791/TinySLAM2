#include <stdio.h>
#include <stdlib.h>

#include <iostream>
#include <thread>

#include <opencv2/opencv.hpp>

#include "CameraDevice.h"
#include "ImageFrame.h"
#include "VisionTracker.h"
#include "Viewer.h"
#include "Timer.h"

using namespace std;

int main(int argc, char** argv)
{

    // camera device 
    CameraIntrinsic K1(argv[1]);
    CameraDevice camera1(K1);
    printf("Camera1: %f %f %f %f %f %f %d %d\n", camera1.K.cx , camera1.K.cy ,camera1.K.fx ,camera1.K.fy ,
            camera1.K.k1 , camera1.K.k2, camera1.K.width ,camera1.K.height ) ;   

    cv::Mat Frame;
    if (!camera1.openCamera(1)) {
        printf("Open camera failed!\n");
        exit(0);
    }
    
    Mapping map(&K1);
    VisionTracker tracker(&K1, &map);                     // Vision Tracker

    Viewer viewer(&map, &tracker);
    std::thread* ptViewer = new std::thread(&Viewer::run, &viewer);

    ImageFrame *refImgFrame = NULL;
    ImageFrame lastFrame;
    bool isFirst = true;
    char cmd = ' ';
    while (true) {

        TIME_BEGIN()

        if ( !camera1.getFrame(Frame, camera1.BGR) ) {
            printf("Get Frame failed!\n");
            break;
        }

        if (cmd == 's') {
            if (isFirst) {              // fisrt 's'
                refImgFrame = new ImageFrame(Frame, &K1); 
                refImgFrame->extractFAST();
                //lastFrame = *refImgFrame;
                map.ClearMap();
                isFirst = false;
            } else {                                // second 's'
                if ( refImgFrame != NULL ) {

                    ImageFrame newImgFrame;
                    
                    TIME_BEGIN()
                    newImgFrame = ImageFrame(Frame, &K1);
                    TIME_END("New Image Frame")

                    // optical flow result
                    newImgFrame.opticalFlowFAST(*refImgFrame);
                    tracker.TrackPose2D2D(*refImgFrame, newImgFrame);
                    map.InitMap(*refImgFrame, newImgFrame);

                    isFirst = true;
                }    
            }
        } else {                                
            if ( refImgFrame != NULL ) {
                ImageFrame newImgFrame;
                    
                TIME_BEGIN()
                newImgFrame = ImageFrame(Frame, &K1);
                TIME_END("New Image Frame")

                // optical flow result
                TIME_BEGIN()
                newImgFrame.opticalFlowFAST(*refImgFrame);
                TIME_END("OpticalFlowFast")

                TIME_BEGIN()
                tracker.TrackPose3D2D(*refImgFrame, newImgFrame);
                TIME_END("TrackPose3D2D")

                for (int i = 0, _end = (int)newImgFrame.trackedPoints.size(); i < _end; i++) { // draw result
                    if (newImgFrame.trackedPoints[i].x > 0) {
                        cv::line(Frame, refImgFrame->points[i], 
                                newImgFrame.trackedPoints[i],
                                cv::Scalar(0, 255, 0));
                    }
                }
            }    
        }

        TIME_END("One Frame")

        cv::imshow("result",Frame);
        cmd = cv::waitKey(33);
    }

    return 0;
}
