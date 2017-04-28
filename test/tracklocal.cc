#include <stdio.h>
#include <stdlib.h>

#include <iostream>
#include <thread>

#include <opencv2/opencv.hpp>

#include "CameraDevice.h"
#include "ImageFrame.h"
#include "Mapping.h"
#include "Initializer.h"
#include "VisionTracker.h"
#include "Viewer.h"
#include "Timer.h"

using namespace std;

int main(int argc, char** argv)
{

    // camera device 
    CameraIntrinsic K1(argv[1]);
    CameraDevice camera1(K1);
    printf("Camera1: %f %f %f %f %f %f %d %d\n", 
            camera1.K.cx , camera1.K.cy ,camera1.K.fx ,camera1.K.fy ,
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
    

    bool started = false;
    char cmd = ' ';
    cv::namedWindow("result");

    Mapping mapping(&K1);
    Initializer initializer(&mapping);
    VisionTracker tracker(&K1, &mapping);
    Viewer viewer(&mapping, &tracker);
    std::thread* ptViewer = new std::thread(&Viewer::run, &viewer);

    ImageFrame *refImgFrame = NULL;
    ImageFrame lastFrame;
    bool isFirst = true;
    bool isInited = false;

    while (true) {

        if ( !camera1.getFrame(Frame, camera1.BGR) ) {
            printf("Get Frame failed!\n");
            break;
        }

        // If is dataset, control at each image
        if (isDataset || isVideo) {
            cmd = cv::waitKey(-1);
        }

        if (cmd == 's') {
            refImgFrame = new ImageFrame(Frame, &K1); 
            tracker.reset();
            tracker.TrackMonocularLocal(*refImgFrame);
            isFirst = true;
        } else {                                
            if ( refImgFrame != NULL ) {

                ImageFrame newImgFrame;

                TIME_BEGIN();
                newImgFrame = ImageFrame(Frame, &K1);
                TIME_END("Construct newImageFrame");

                tracker.TrackMonocularLocal(newImgFrame);

                // draw optical flow
                for (Measure2D m2d:newImgFrame.measure2ds) { // draw result
                    if (m2d.valid && m2d.ref2d->valid) {
                        cv::line(Frame, 
                                m2d.pt, 
                                m2d.ref2d->pt,
                                cv::Scalar(0, 255, 0),
                                1, cv::LINE_AA);
                    }
                }

            }    
        }

        cv::imshow("result",Frame);
        cmd = cv::waitKey(33);
    }

    return 0;
}
