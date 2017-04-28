#ifndef IMAGEFRAME_H
#define IMAGEFRAME_H

#include <stdio.h>
#include <stdlib.h>

#include <iostream>
#include <mutex>

#include <boost/bimap.hpp>

#include <opencv2/opencv.hpp>
#include <opencv2/video/tracking.hpp>
#include <opencv2/features2d/features2d.hpp>

#include <TooN/se3.h>
#include <TooN/so3.h>

//#include <cvd/image_io.h>
//#include <cvd/vision.h>
//#include <cvd/esm.h>

#include "CameraIntrinsic.h"

using namespace std;

//typedef boost::bimap< int, cv::Point3f* > Map_2d_3d;
//typedef Map_2d_3d::value_type Map_2d_3d_key_val;

class Measure3D;
class ImageFrame;

class ImageLevel 
{
    public:
        int levelIdx;
        double scaleFactor;

        cv::Mat image;
        vector< cv::Point2f > points;

        ImageLevel(cv::Mat& im, int idx, double scale);
};

class Measure2D
{
    public:

        ImageFrame *refFrame;
        cv::Point2f pt;
        cv::Point2f undisPt;
        int levelIdx;

        int outlierNum;
        bool valid;

        Measure3D* ref3d;
        Measure2D* ref2d;

        Measure2D() = default;
        Measure2D(cv::Point2f& _pt, 
                cv::Point2f& _undisPt, 
                ImageFrame* _refFrame,
                int _levelIdx)
            :refFrame(_refFrame),
            pt(_pt), 
            undisPt(_undisPt),
            levelIdx(_levelIdx),
            outlierNum(0),
            valid(true),
            ref3d(NULL),
            ref2d(this){};
};

class ImageFrame
{
    public:
        vector< ImageLevel > levels;
        vector< Measure2D > measure2ds;

        cv::Mat descriptors;

//        Map_2d_3d map_2d_3d;

        TooN::SE3<> mTcw;                      // Transformation 
                                               // from w to camera
                                               
        cv::Mat R, t;                          // Tcw R, t
        ImageFrame* mRefFrame;                 // Reference Frame
        CameraIntrinsic* K;                    // CameraIntrinsic

        bool isKeyFrame;                       // only used for   
                                               // key frame
        bool isSelfReference;

        ImageFrame() = default;
        explicit ImageFrame(const cv::Mat& frame, CameraIntrinsic* _K);
        explicit ImageFrame(const ImageFrame& imgFrame);

        void operator=(const ImageFrame& imgFrame);

        void extractFAST(int lowNum = 400, int highNum = 500);
        int opticalFlowFAST(ImageFrame& refFrame);
        int opticalFlowFASTAndValidate(ImageFrame& refFrame);
        void opticalFlowTrackedFAST(ImageFrame& lastFrame);

        void extractPatch();
        cv::Mat extractTrackedPatch();
        void computePatchDescriptorAtPoint(
                const cv::Point2f &pt, 
                const cv::Mat &im,
                float* desc);
        void trackPatchFAST(ImageFrame& refFrame);

        vector< int > fuseFAST();

        //void SBITrackFAST(ImageFrame& refFrame);

        cv::Mat GetTwcMat();
        cv::Mat GetTcwMat();

};

#endif
