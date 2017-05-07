#ifndef INITIALIZER_H
#define INITIALIZER_H

#include <stdio.h>
#include <stdlib.h>

#include <iostream>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/video/tracking.hpp>
#include <opencv2/imgproc.hpp>

#include <g2o/core/base_vertex.h>
#include <g2o/core/base_unary_edge.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/robust_kernel.h>
#include <g2o/core/robust_kernel_impl.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/solvers/csparse/linear_solver_csparse.h>
#include <g2o/solvers/dense/linear_solver_dense.h>
#include <g2o/types/sba/types_six_dof_expmap.h>


#include "ImageFrame.h"
#include "Mapping.h"
#include "Timer.h"
#include "Converter.h"

using namespace std;

class Initializer
{
    public:

        enum State {
            NOTINITIALIZED,
            INITIALIZING,
            INITIALIZED
        };

        State state;

        ImageFrame *firstFrame;
        ImageFrame lastFrame;

        Mapping* map;

        bool isFisrtFellow;

        ImageFrame *k1, *k2;

        Initializer() = default;
        Initializer(Mapping *_map):
            state(NOTINITIALIZED),
            firstFrame(NULL),
            map(_map),
            isFisrtFellow(true){};
        ~Initializer() = default;

        void SetFirstFrame(ImageFrame *f);
        bool TrackFeatureAndCheck(ImageFrame *f);

        bool TryInitialize(ImageFrame *f);
        bool RobustTrackPose2D2D(ImageFrame &lf, 
                ImageFrame &rf);

        bool TryInitializeByThirdParty(ImageFrame *f);

        bool TryInitializeByG2O(ImageFrame *f);
        bool RobustTrackPose2D2DG2O(ImageFrame &lf, 
                ImageFrame &rf);

        bool CheckPoints(cv::Mat &R, cv::Mat &t, cv::Mat &pts);
};

#endif
