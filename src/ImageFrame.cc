#include "ImageFrame.h"
#include "Timer.h"
#include "Converter.h"

ImageLevel::ImageLevel(cv::Mat& im, int idx, double scale):
        levelIdx(idx), 
        scaleFactor(scale), 
        image(im.clone()) 
{

}

ImageFrame::ImageFrame(const cv::Mat& frame, CameraIntrinsic* _K): 
    //SBI(CVD::ImageRef(32, 24)), 
    R(cv::Mat::eye(3,3, CV_64FC1)), 
    t(cv::Mat::zeros(3,1, CV_64FC1)), 
    mRefFrame(this), K(_K), 
    isKeyFrame(false),
    isSelfReference(false)
{
    cv::Mat image;

    if (frame.channels()==3) {
        cv::cvtColor(frame, image, CV_BGR2GRAY);
    } else if (frame.channels() == 4) {
        cv::cvtColor(frame, image, CV_BGRA2GRAY);
    } else {
        frame.copyTo(image);
    }

    //cv::Mat s_img;
    //cv::resize(image, s_img, cv::Size(32, 24));
    //cv::blur(s_img, s_img, cv::Size(3,3));

    //memcpy(SBI.begin(), s_img.data, 32*24*sizeof(unsigned char));
    
    levels.reserve(4);

    // level 0
    levels.push_back(ImageLevel(image, 0, 1.0));

    // level 1
    cv::Mat image1;
    cv::pyrDown(image, image1, cv::Size(image.cols/2., image.rows/2.));
    levels.push_back(ImageLevel(image1, 1, 2.));

    // level 2
    cv::Mat image2;
    cv::pyrDown(image1, image2, cv::Size(image1.cols/2., image1.rows/2.));
    levels.push_back(ImageLevel(image2, 2, 4.));

    // level 3
    cv::Mat image3;
    cv::pyrDown(image2, image3, cv::Size(image2.cols/2., image2.rows/2.));
    levels.push_back(ImageLevel(image3, 3, 8.));
}

ImageFrame::ImageFrame(const ImageFrame& imgFrame): 
    levels(imgFrame.levels), 
    //SBI(imgFrame.SBI), 
    measure2ds(imgFrame.measure2ds), 
    descriptors(imgFrame.descriptors), 
    //map_2d_3d(imgFrame.map_2d_3d),
    mTcw(imgFrame.mTcw), 
    R(imgFrame.R.clone()), 
    t(imgFrame.t.clone()),
    mRefFrame(imgFrame.mRefFrame), 
    K(imgFrame.K), 
    isKeyFrame(imgFrame.isKeyFrame),
    isSelfReference(imgFrame.isSelfReference)
{
    for (int i = 0, _end = (int)measure2ds.size(); 
            i < _end; i++ ) {
        measure2ds[i].refFrame = this;
        if (isSelfReference)
            measure2ds[i].ref2d = &(measure2ds[i]);
    }
}

void ImageFrame::operator=(const ImageFrame& imgFrame)
{
    levels = imgFrame.levels;
    measure2ds = imgFrame.measure2ds;
    descriptors = imgFrame.descriptors.clone();

    mTcw = imgFrame.mTcw;

    R = imgFrame.R.clone();
    t = imgFrame.t.clone();

    mRefFrame = imgFrame.mRefFrame;
    K = imgFrame.K;

    isKeyFrame = imgFrame.isKeyFrame;
    isSelfReference = imgFrame.isSelfReference;

    for (int i = 0, _end = (int)measure2ds.size(); 
            i < _end; i++ ) {
        measure2ds[i].refFrame = this;
        if (isSelfReference)
            measure2ds[i].ref2d = &(measure2ds[i]);
    }
}

void ImageFrame::extractFAST(int lowNum, int highNum)
{
    int thres = 40;
    int low = lowNum;
    int high = highNum;

    // extract FAST for each level
    for (int i = 0; i < 4; i++) {
        vector< cv::KeyPoint > keyPoints;

        cv::Rect ROI(3,3, 
                levels[i].image.cols-7, 
                levels[i].image.rows-7);

        if (i == 0) {
            int iter = 0;
            while((int)keyPoints.size() < low 
                      || (int)keyPoints.size() > high)
            {
                if ((int)keyPoints.size() < low) {
                    thres *= 0.9;
                } else if ((int)keyPoints.size() > high) {
                    thres *= 1.1;
                }
                cv::FAST(cv::Mat(levels[i].image,ROI), 
                        keyPoints, thres, 
                        true, cv::FastFeatureDetector::TYPE_9_16);
                printf("extract keyPoints: %d\n", 
                        keyPoints.size());
                iter++;
                if (iter > 20) {
                    printf("error: cannot extract suitable FAST!\n");
                    exit(0);
                }
            }
        } else {
            cv::FAST(cv::Mat(levels[i].image, ROI), 
                    keyPoints, thres, 
                    true, cv::FastFeatureDetector::TYPE_9_16);
        }

        levels[i].points.reserve(keyPoints.size());
        levels[i].points.resize(0);
        for (int j = 0, _end = (int)keyPoints.size(); 
                j < _end; j++) {
            levels[i].points.push_back( keyPoints[j].pt );
        }
        printf("level %d extract %d FAST features\n", 
                i, keyPoints.size());
    }

    // set measure2ds
    measure2ds.reserve(levels[0].points.size()+
            levels[1].points.size()+
            levels[2].points.size()+
            levels[3].points.size());
    measure2ds.resize(0);
    for (int i = 0; i < 4; i++) {
        for (int j = 0, _end = (int)levels[i].points.size(); 
                j < _end; j++) {
            cv::Point2f pt(
                    levels[i].points[j].x*levels[i].scaleFactor, 
                    levels[i].points[j].y*levels[i].scaleFactor);
            cv::Point2f undisPt = K->undistort(pt.x, pt.y);
            measure2ds.push_back( Measure2D(pt, undisPt, this, i) );
        }
    }

    for (int i = 0, _end = (int)measure2ds.size(); i < _end; i++) {
        measure2ds[i].ref2d = &(measure2ds[i]);
    }

    isSelfReference = true;
    
}

vector< int > ImageFrame::fuseFAST()
{

    return vector< int > ();
}

void ImageFrame::extractPatch()
{
    //if (points.size() == 0) {
    //    return;
    //}

    //descriptors.create(points.size(), 49, CV_32FC1);
    //for (int i = 0, _end = (int)points.size(); i < _end; i++) {
    //    computePatchDescriptorAtPoint(points[i], image,
    //            descriptors.ptr<float>(i));
    //}
}

cv::Mat ImageFrame::extractTrackedPatch()
{
    //if (trackedPoints.size() == 0) {
    //    return cv::Mat();
    //}

    //cv::Mat res(trackedPoints.size(), 49, CV_32FC1);
    //for (int i = 0, _end = (int)trackedPoints.size(); i < _end; i++) {
    //    computePatchDescriptorAtPoint(trackedPoints[i], image,
    //            res.ptr<float>(i));
    //}

    //return res;
    
    return cv::Mat();
}

void ImageFrame::computePatchDescriptorAtPoint(const cv::Point2f &pt, 
        const cv::Mat &im, float* desc)
{
    const int PATCH_SIZE = 7;
    const int HALF_PATCH_SIZE = 3; 

    if (pt.x < HALF_PATCH_SIZE || pt.x > im.cols - HALF_PATCH_SIZE
            || pt.y < HALF_PATCH_SIZE || pt.y > im.rows - HALF_PATCH_SIZE) {
        return;
    }

    int centerIdx = pt.x + pt.y*im.cols;
    int startIdx = centerIdx - HALF_PATCH_SIZE*im.cols - HALF_PATCH_SIZE;

    const unsigned char* data = im.data;

    // Ave
    float aveVal = 0; 
    int nowIdx=startIdx;
    for (int r = 0; r < PATCH_SIZE; r++,nowIdx += im.cols) {
        for (int c = 0; c < PATCH_SIZE; c++,nowIdx++) {
            aveVal += float(data[nowIdx]);
        }
    }
    aveVal =  aveVal / (PATCH_SIZE*PATCH_SIZE);

    // Devi
    float devi = 0;
    nowIdx = startIdx;
    for (int r = 0; r < PATCH_SIZE; r++,nowIdx += im.cols) {
        for (int c = 0; c < PATCH_SIZE; c++,nowIdx++) {
            float val = float(data[nowIdx]);
            devi += (val - aveVal)*(val - aveVal);
        }
    }
    devi /= (PATCH_SIZE*PATCH_SIZE);
    devi = sqrt(devi);

    // Desc
    int desIdx = 0;
    nowIdx = startIdx;
    for (int r = 0; r < PATCH_SIZE; r++,nowIdx += im.cols) {
        for (int c = 0; c < PATCH_SIZE; c++,nowIdx++,desIdx++) {
            desc[desIdx] = (float(data[nowIdx]) - aveVal) / devi;
        }
    }
    
}

void ImageFrame::trackPatchFAST(ImageFrame& refFrame)
{

//    mRefFrame = &refFrame; 
//
//    // extract FAST
//    TIME_BEGIN()
//    extractFAST();
//    TIME_END("FAST time: ")
//
//    // extractPatch
//    TIME_BEGIN()
//    extractPatch();
//    TIME_END("Extract Patch time: ")
//
//    if ( descriptors.empty() || descriptors.rows == 0)
//        return;
//
//    // match
//    cv::BFMatcher matcher(cv::NORM_L2, true);
//    vector< cv::DMatch > matches;
//    TIME_BEGIN()
//    matcher.match(refFrame.descriptors, descriptors, matches);
//    TIME_END("Match time: ")
//
//    cout << "find matches: " << matches.size() << endl;
//
//    trackedPoints.resize(refFrame.points.size());
//    undisTrackedPoints.resize(refFrame.points.size());
//
//    std::fill(trackedPoints.begin(), trackedPoints.end(), 
//            cv::Point2f(0, 0));
//    std::fill(undisTrackedPoints.begin(), undisTrackedPoints.end(), 
//            cv::Point2f(0, 0));
//
//    float maxDis = 0, minDis = 999;
//    for (int i = 0, _end = (int)matches.size(); i < _end; i++) {
//       if (matches[i].distance > maxDis)
//           maxDis = matches[i].distance;
//       if (matches[i].distance < minDis)
//           minDis = matches[i].distance;
//    }
//    float thres = minDis + (maxDis - minDis) * 0.2;
//    
//    for (int i = 0, _end = (int)matches.size(); i < _end; i++) {
//        if (matches[i].distance < thres)
//        {
//            trackedPoints[matches[i].queryIdx] = points[matches[i].trainIdx];
//            undisTrackedPoints[matches[i].queryIdx] = 
//                K->undistort(points[matches[i].trainIdx].x,
//                        points[matches[i].trainIdx].y);
//        }
//    }
//
//    cout << " track patch FAST done." << endl;
}

int ImageFrame::opticalFlowFAST(ImageFrame& refFrame)
{

    // prepare points
    vector< cv::Point2f > pt_1, pt_2, undispt_2;
    pt_1.reserve(refFrame.measure2ds.size());
    for (int i = 0, _end = (int)refFrame.measure2ds.size(); 
            i < _end; i++ ) {
        if (refFrame.measure2ds[i].valid)
            pt_1.push_back(refFrame.measure2ds[i].pt);
    }

    // optical flow points
    cv::Mat status, err;
    TIME_BEGIN();
    cv::calcOpticalFlowPyrLK(
            refFrame.levels[0].image, 
            levels[0].image, 
            pt_1, pt_2, 
            status, err, 
            cv::Size(15 ,15),
            3);
            //);
    TIME_END("calcOpticalFlowPyrLK");
    undispt_2.reserve(pt_2.size());
    for (int i = 0, _end = (int)pt_2.size(); i < _end; i++ ) {
        undispt_2.push_back(K->undistort(pt_2[i].x, pt_2[i].y));
    }

    //// check essential matrix, validation
    //vector< cv::Point2f > org_pt;
    //org_pt.reserve(undispt_2.size());
    //for (int i = 0, _end = (int)pt_1.size(); i < _end; i++) {
    //    org_pt.push_back(refFrame.measure2ds[i].ref2d->undisPt);
    //}

    //cv::Mat inlier;
    //TIME_BEGIN();
    //cv::findEssentialMat(org_pt, undispt_2, 
    //        (K->fx + K->fy)/2, 
    //        cv::Point2d(K->cx, K->cy),
    //        cv::RANSAC, 0.999, 3, inlier);
    //TIME_END("essential matrix estimation");

    //// set measures2ds
    //measure2ds.reserve(pt_2.size());
    //for (int i = 0, _end = (int)undispt_2.size(); i < _end; i++) {
    //    if (inlier.at<unsigned char>(i) == 0) {
    //        refFrame.measure2ds[i].ref2d->outlierNum++;
    //        if (refFrame.measure2ds[i].ref2d->outlierNum > 3) {
    //            refFrame.measure2ds[i].ref2d->valid = false;
    //        } else {
    //            Measure2D measure2d(pt_2[i], undispt_2[i], -1);
    //            measure2d.ref2d = refFrame.measure2ds[i].ref2d;
    //            measure2ds.push_back(measure2d);    
    //        }
    //    } else {
    //        Measure2D measure2d(pt_2[i], undispt_2[i], -1);
    //        measure2d.ref2d = refFrame.measure2ds[i].ref2d;
    //        measure2ds.push_back(measure2d);
    //    }
    //}
    
    // set measure2ds
    measure2ds.reserve(pt_2.size());
    for (int i = 0, _end = (int)pt_2.size(); i < _end; i++) {
        if (status.at<unsigned char>(i) == 0) {
            refFrame.measure2ds[i].ref2d->outlierNum++;
            if (refFrame.measure2ds[i].ref2d->outlierNum > 3) {
                refFrame.measure2ds[i].ref2d->valid = false;
            }
        } else {
            Measure2D measure2d(pt_2[i], undispt_2[i], this, refFrame.measure2ds[i].ref2d->levelIdx);
            measure2d.ref2d = refFrame.measure2ds[i].ref2d;
            measure2ds.push_back(measure2d);    
        }
    }

    printf("Optical flow on %d points\n", measure2ds.size());

    return measure2ds.size();
}

int ImageFrame::opticalFlowFASTAndValidate(ImageFrame& refFrame)
{
    return 0;
}

void ImageFrame::opticalFlowTrackedFAST(ImageFrame& lastFrame)
{
//    mRefFrame = &lastFrame;
//
//    vector< cv::Point2f > pts, undis_pts, undis_flow_pts,flow_pts;
//    vector< int > idxs;
//    pts.reserve(lastFrame.trackedPoints.size());
//    undis_pts.reserve(lastFrame.trackedPoints.size());
//    flow_pts.reserve(lastFrame.trackedPoints.size());
//    undis_flow_pts.reserve(lastFrame.trackedPoints.size());
//    idxs.reserve(lastFrame.trackedPoints.size());
//
//    for (int i = 0, _end = (int)lastFrame.trackedPoints.size(); 
//            i < _end; i++) {
//        if (lastFrame.trackedPoints[i].x > 0) {
//            pts.push_back(lastFrame.trackedPoints[i]);
//            undis_pts.push_back(lastFrame.undisTrackedPoints[i]);
//            idxs.push_back(i);
//        }
//    }
//
//    cv::Mat status, err;
//    //TIME_BEGIN()
//    cv::calcOpticalFlowPyrLK(lastFrame.image, image, 
//            pts, flow_pts, status, err) ;
//    //TIME_END("Optical Flow")
//
//    for (int i = 0, _end = (int)flow_pts.size(); i < _end; i++) {
//        undis_flow_pts.push_back( 
//                K->undistort(flow_pts[i].x, flow_pts[i].y) );
//    }
//
//    // essential matrix estimation validation
//    //cv::Mat inlier;
//    //TIME_BEGIN()
//    //cv::findEssentialMat(undis_pts, undis_flow_pts, 
//    //        (K->fx + K->fy)/2, cv::Point2d(K->cx, K->cy),
//    //        cv::RANSAC, 0.9999, 2, inlier);
//    //TIME_END("essential matrix estimation")
//
//    trackedPoints.resize(lastFrame.trackedPoints.size());
//    undisTrackedPoints.resize(lastFrame.trackedPoints.size());
//    fill(trackedPoints.begin(), 
//            trackedPoints.end(), cv::Point2f(0,0));
//    fill(undisTrackedPoints.begin(), 
//            undisTrackedPoints.end(), cv::Point2f(0,0));
//    for (int i = 0, _end = (int)undis_flow_pts.size(); i < _end; i++) {
//        //if (inlier.at<unsigned char>(i) == 1) {
//          trackedPoints[idxs[i]] = flow_pts[i];
//          undisTrackedPoints[idxs[i]] = undis_flow_pts[i];
//        //} else {
//
//        //}
//    }

}

//void ImageFrame::SBITrackFAST(ImageFrame& refFrame)
//{
//    mRefFrame = &refFrame;
//
//    CVD::Homography<8> homography;
//    CVD::StaticAppearance appearance;
//    CVD::Image< TooN::Vector<2> > greImg 
//            = CVD::Internal::gradient<TooN::Vector<2>, unsigned char>(refFrame.SBI);
//    CVD::Internal::esm_opt(homography, appearance, refFrame.SBI, greImg, SBI, 40, 1e-8, 1.0);
//    TooN::Matrix<3> H = homography.get_matrix();
//
//
//    H(0,2) = H(0,2) * 20.f;
//    H(1,2) = H(1,2) * 20.f;
//    H(2,0) = H(2,0) / 20.f;
//    H(2,1) = H(2,1) / 20.f;
//
//    keyPoints.resize(0);
//
//    for (int i = 0, _end = (int)refFrame.keyPoints.size(); i < _end; i++ ) {
//        TooN::Vector<3> P;
//        P[0] = refFrame.keyPoints[i].pt.x;
//        P[1] = refFrame.keyPoints[i].pt.y;
//        P[2] = 1;
//        TooN::Vector<3> n_P = H * P;
//        keyPoints.push_back(cv::KeyPoint(n_P[0]/n_P[2], n_P[1]/n_P[2], 10));
//    }
//
//}

cv::Mat ImageFrame::GetTwcMat()
{
    if (R.empty() || t.empty())
        return cv::Mat();

    cv::Mat res = cv::Mat::eye(4, 4, CV_64FC1);
    R.copyTo(res.rowRange(0,3).colRange(0,3));
    t.copyTo(res.rowRange(0,3).col(3));
    return res;
}

cv::Mat ImageFrame::GetTcwMat()
{
    if (R.empty() || t.empty() )
        return cv::Mat();

    cv::Mat res = cv::Mat::eye(4, 4, CV_64FC1);
    cv::Mat Rt = R.t();
    cv::Mat _t = -Rt*t;
    Rt.copyTo(res.rowRange(0,3).colRange(0,3));
    _t.copyTo(res.rowRange(0,3).col(3));
    return res;
}
