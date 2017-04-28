#ifndef CONVERTER_H
#define CONVERTER_H

#include <stdio.h>
#include <stdlib.h>

#include <iostream>
#include <sstream>

#include <opencv2/opencv.hpp>

#include <TooN/se3.h>
#include <TooN/so3.h>
#include <TooN/TooN.h>

using namespace std;

namespace Converter
{
    
    void TooNSO3_Mat(const TooN::SO3<>& so3, cv::Mat& mat );
    void Mat_TooNSO3(const cv::Mat& mat, TooN::SO3<> & so3);
    cv::Point3f TransfromPoint3D(
            const cv::Point3f& P, const cv::Mat& R, const cv::Mat& t);
    std::string getImageType(int number);

}

#endif
