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
    

    bool started = false;
    char cmd = ' ';
    cv::namedWindow("result");

    ImageFrame *refImgFrame = NULL;
    ImageFrame lastFrame;
    bool isFirst = true;

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
            refImgFrame->extractFAST();
            refImgFrame->setFASTAsMeasure();
            lastFrame = *refImgFrame;
            isFirst = true;
        } else {                                
            if ( refImgFrame != NULL ) {

                ImageFrame newImgFrame;
                TIME_BEGIN();
                newImgFrame = ImageFrame(Frame, &K1);
                TIME_END("Construct newImageFrame");

                // optical flow result
                TIME_BEGIN();
                if (isFirst) {
                    newImgFrame.opticalFlowMeasure(*refImgFrame);
                    isFirst = false;
                } else {
                    newImgFrame.opticalFlowMeasure(lastFrame);
                }
                TIME_END("OpticalFlowTrackedFast");

                lastFrame = newImgFrame;

                for (int i = 0, _end = 
                        (int)newImgFrame.measure2ds.size(); 
                        i < _end; i++) { // draw result
                    cv::line(Frame, 
                            newImgFrame.measure2ds[i].ref2d->pt, 
                            newImgFrame.measure2ds[i].pt,
                            cv::Scalar(0, 255, 0));
                }


                printf("Track inlier ratio: %f\n", 
                        (double)newImgFrame.measure2ds.size()/
                        (double)refImgFrame->measure2ds.size());

                vector<int> sum(4, 0);
                vector<int> valid(4, 0);

                for (int i = 0, _end = 
                        (int)refImgFrame->measure2ds.size(); 
                        i < _end; i++) {
                    sum[refImgFrame->measure2ds[i].levelIdx]++;
                    if (refImgFrame->measure2ds[i].valid)
                        valid[refImgFrame->measure2ds[i].
                                levelIdx]++;
                }

                for (int i = 0; i < 4; i++) {
                    printf("level %d valid ratio %f\n", i, 
                            (double)valid[i] / (double)sum[i]);
                }
            }    
        }

        cv::imshow("result",Frame);
        cmd = cv::waitKey(33);
    }

    return 0;
}
