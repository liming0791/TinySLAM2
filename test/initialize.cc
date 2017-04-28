#include <stdio.h>
#include <stdlib.h>

#include <iostream>
#include <thread>

#include <opencv2/opencv.hpp>

#include "CameraDevice.h"
#include "ImageFrame.h"
#include "Mapping.h"
#include "Initializer.h"
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
    Viewer viewer(&mapping, NULL);
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
            refImgFrame->extractFAST();
            initializer.SetFirstFrame(refImgFrame);
            isFirst = true;
        } else {                                
            if ( refImgFrame != NULL ) {

                ImageFrame newImgFrame;

                TIME_BEGIN();
                newImgFrame = ImageFrame(Frame, &K1);
                TIME_END("Construct newImageFrame");

                if (!isInited) {
                    isInited = initializer.TryInitialize(&newImgFrame);
                    // draw optical flow
                    for (int i = 0, _end = 
                            (int)newImgFrame.measure2ds.size(); 
                            i < _end; i++) { // draw result
                        cv::line(Frame, 
                                newImgFrame.measure2ds[i].ref2d->pt, 
                                newImgFrame.measure2ds[i].pt,
                                cv::Scalar(0, 255, 0),
                                1, cv::LINE_AA);
                    }


                    printf("Track inlier ratio: %f\n", 
                            (double)newImgFrame.measure2ds.size()/
                            (double)initializer.firstFrame->measure2ds.size());

                    vector<int> sum(4, 0);
                    vector<int> valid(4, 0);

                    for (int i = 0, _end = 
                            (int)initializer.firstFrame->measure2ds.size(); 
                            i < _end; i++) {
                        sum[initializer.firstFrame->measure2ds[i].levelIdx]++;
                        if (initializer.firstFrame->measure2ds[i].valid)
                            valid[initializer.firstFrame->measure2ds[i].
                                levelIdx]++;
                    }

                    for (int i = 0; i < 4; i++) {
                        printf("level %d valid ratio %f\n", i, 
                                (double)valid[i] / (double)sum[i]);
                    }

                    cv::imshow("result",Frame);

                } else if (isInited) {
                    ImageFrame *k1 = initializer.k1;
                    ImageFrame *k2 = initializer.k2;

                    vector<cv::KeyPoint> pt1, pt2;
                    vector<cv::DMatch> matches;

                    for (int i = 0, _end = (int)k2->measure2ds.size(); i < _end; i++) {
                        if (k2->measure2ds[i].valid) {
                            pt1.push_back(cv::KeyPoint(k2->measure2ds[i].ref2d->pt, 3));
                            pt2.push_back(cv::KeyPoint(k2->measure2ds[i].pt, 3));
                            matches.push_back(cv::DMatch(pt1.size()-1, pt1.size()-1, 0));
                        }
                    }

                    printf("pt1 size: %d, pt2 size: %d\n", pt1.size(), pt2.size());

                    cv::Mat result;
                    cv::drawMatches(k1->levels[0].image, pt1, k2->levels[0].image, pt2, 
                            matches, result);

                    cv::imshow("result", result);
                    
                }
            }    
        }

        cmd = cv::waitKey(33);
    }

    return 0;
}
