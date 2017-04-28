#include "CameraDevice.h"

CameraDevice::CameraDevice(const CameraIntrinsic& _K) 
{
    SetIntrinsic(_K);
    SetFPS(30);
}

void CameraDevice::SetIntrinsic(const CameraIntrinsic& _K)
{
    K = _K;
}

void CameraDevice::SetFPS(int _FPS)
{
    FPS = _FPS;
}

bool CameraDevice::openCamera(int id)
{
    video.set(CV_CAP_PROP_FPS, FPS);
    if ( !video.open(id) ) {
        printf("Open camera %d failed!\n", id);
        return false;
    }
    sourceType = CAMERA;
    return true;
}

bool CameraDevice::openVideo(const string& filename)
{
    if ( !video.open(filename) ) {
        printf("Open video %s failed!\n", filename.c_str());
        return false;
    }
    sourceType = VIDEO;
    return true;
}

bool CameraDevice::openDataset(const string& filename)
{
    file.open(filename);
    if ( !file.is_open() ) {
        printf("Open dataset %s failed!\n", filename.c_str());
        return false;
    }

    // test read the first image
    if (getline(file, frameName) ) {
        Frame = cv::imread(frameName);
        if (Frame.empty()) {
            printf("Open dataset %s failed! Test first image %s failed!\n", filename.c_str(), frameName.c_str());
            return false;
        }
        file.clear();
        file.seekg(0, ios::beg);
    } else {
        printf("Open dataset %s failed! Get line failed!\n", filename.c_str());
        return false;
    }

    sourceType = DATASET;
    return true;
}

bool CameraDevice::getFrame(cv::Mat& frame, FrameType type)
{
    if (sourceType == DATASET) {
        if (getline(file, frameName)) {
            Frame = cv::imread(frameName);
            if (Frame.empty()) {
                printf("Read image %s failed!\n", frameName.c_str());
                return false;
            }
        } else {
            printf("Reach end of the dataset!\n");
            return false;
        }
    } else if (sourceType == VIDEO) {
        if (!video.read(Frame)) {
            printf("Reach end of the video!\n");
            return false;
        }
    } else if (sourceType == CAMERA) {
        video >> Frame;
    }

    if ( type == GRAY ) {
        if ( Frame.channels() == 3 ) {
            cv::cvtColor(Frame, frame, CV_BGR2GRAY);
        } else if ( Frame.channels() == 4 ) {
            cv::cvtColor(Frame, frame, CV_BGRA2GRAY);
        } else {
            Frame.copyTo(frame);
        }
    } else {
        if ( Frame.channels() == 3 ) {
            Frame.copyTo(frame);
        } else if ( Frame.channels() == 4 ) {
            cv::cvtColor(Frame, frame, CV_BGRA2BGR);
        } else {
            Frame.copyTo(frame);
            printf("Frame data may be gray scale, no bgr data!\n");
            return false;
        }
    }

    return true;
}

