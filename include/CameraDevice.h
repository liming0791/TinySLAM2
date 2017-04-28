#ifndef CAMERADEVICE_H
#define CAMERADEVICE_H

#include <stdio.h>
#include <stdlib.h>

#include <iostream>
#include <fstream>
#include <sstream>

#include <opencv2/opencv.hpp>

#include "CameraIntrinsic.h"

using namespace std;

class CameraDevice 
{
    public:

        enum FrameType {
            GRAY,
            BGR
        } ;

        enum SourceType {
            CAMERA,
            VIDEO,
            DATASET
        } ;

        CameraIntrinsic K;
        int FPS;
        
        CameraDevice() = default;
        explicit CameraDevice(const CameraIntrinsic& _K);
        void SetIntrinsic(const CameraIntrinsic& _K);
        void SetFPS(int _FPS);

        bool openCamera(int id);
        bool openVideo(const string& filename);
        bool openDataset(const string& filename);
        bool getFrame(cv::Mat& frame, FrameType type);

    private:
        // for camera and video file
        cv::VideoCapture video;
        // for dataset file
        ifstream file;
        string frameName;
        // image data
        cv::Mat Frame;

        SourceType sourceType;
     
};

#endif
