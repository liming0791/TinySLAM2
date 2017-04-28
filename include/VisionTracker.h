#ifndef VISIONTRACKER_H
#define VISIONTRACKER_H

#include <stdio.h>
#include <stdlib.h>

#include <iostream>
#include <sstream>
#include <chrono>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/video/tracking.hpp>
#include <opencv2/imgproc.hpp>

#include <TooN/se3.h>
#include <TooN/so3.h>

#include <g2o/core/base_vertex.h>
#include <g2o/core/base_unary_edge.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/robust_kernel.h>
#include <g2o/core/robust_kernel_impl.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/solvers/csparse/linear_solver_csparse.h>
#include <g2o/solvers/dense/linear_solver_dense.h>
#include <g2o/types/sba/types_six_dof_expmap.h>

#include "CameraIntrinsic.h"
#include "ImageFrame.h"
#include "MedianFilter.h"
#include "Mapping.h"
#include "Initializer.h"

using namespace std;

class VisionTracker
{
    public:

        enum TrackingStatus {
            NOTINITIALIZED,
            INITIALIZING,
            INITIALIZED
        };

        TrackingStatus state;

        CameraIntrinsic* K;
        Mapping* map;

        TooN::SE3<> mTcwNow;
        cv::Mat mR, mt;

        ImageFrame* refFrame;
        ImageFrame lastFrame;
        bool isFirstFellow;

        int countFromLastKeyFrame;

        VisionTracker() = default;
        VisionTracker(CameraIntrinsic* _K, Mapping* _map):
            state(NOTINITIALIZED), 
            K(_K), 
            map(_map), 
            isFirstFellow(true),
            countFromLastKeyFrame(0),
            initializer(_map)
            { };

        void TrackMonocular(ImageFrame& f);
        void TrackMonocularLocal(ImageFrame& f);
        //void TrackMonocularNewKeyFrame(ImageFrame& f);
        //void TryInitialize(ImageFrame& f);
        //void TryInitializeByG2O(ImageFrame& f);
        void reset();

        void TrackByRefFrame(ImageFrame& ref, ImageFrame& f);
        //void InsertKeyFrame(ImageFrame& kf, ImageFrame& f);
        //void TrackPose2D2D(const ImageFrame& lf, ImageFrame& rf );
        //void TrackPose2D2DG2O(ImageFrame& lf, ImageFrame& rf );
        //void TrackPose3D2D(const ImageFrame& lf, ImageFrame& rf );
        //void TriangulateNewPoints(ImageFrame&lf, ImageFrame& rf );
        //void TrackPose3D2DHybrid(ImageFrame& lf, ImageFrame& rf );

        //double TrackFeatureOpticalFlow(ImageFrame& kf, ImageFrame& f);
        //void updateRefFrame(ImageFrame* kf);

        cv::Mat GetTwcMatNow();
        cv::Mat GetTcwMatNow();

        MedianFilter<5> medianFilter[6];

    private:
        void BA3D2D(const vector< cv::Point3f > & points_3d,
                const vector< cv::Point2f > & points_2d,
                cv::Mat& R, 
                cv::Mat& t);

        Initializer initializer;

        std::mutex mRefFrameMutex;
};

#endif
