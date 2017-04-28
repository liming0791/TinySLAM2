#ifndef CAMERAINTRINSIC_H
#define CAMERAINTRINSIC_H

#include <stdio.h>
#include <stdlib.h>

#include <iostream>
#include <fstream>
#include <sstream>

#include <opencv2/opencv.hpp>

using namespace std;

class CameraIntrinsic 
{
    public:
        double cx;
        double cy;
        double fx;
        double fy;
        double k1;
        double k2;

        int width;
        int height;

        CameraIntrinsic() = default;
        explicit CameraIntrinsic(const string& fileName);
        CameraIntrinsic(double _cx, double _cy, double _fx, double _fy, double _k1, double _k2,
                int _width, int _height);
        void loadFromFile(const string& fileName);

        cv::Point2f distort(int x, int y);
        cv::Point2f undistort(int x, int y);

        cv::Point2f pixel2device(float x, float y);
        cv::Point2f device2pixel(float u, float v);

        cv::Point3f Proj2Dto3D(float x, float y, float d);
        cv::Point2f Proj3Dto2D(float x, float y, float z);
};

#endif
