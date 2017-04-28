#include "Mapping.h"
//#include "VisionTracker.h"

Measure3D::Measure3D(const cv::Point3f &p)
    :pt(p), outlierNum(0), valid(true)
{

}

//void Mapping::SetTracker(VisionTracker* _tracker)
//{
//    tracker = _tracker;   
//}

void Mapping::Run()
{
//    ImageFrame nowFrame;
//    bool hasFrame = false;
//
//    while (1) {
//        
//        {   // lock queue
//            std::lock_guard<std::mutex> lock(mQueueMutex);
//            if (!keyFrameQueue.empty()) {
//                nowFrame = keyFrameQueue.front();
//                keyFrameQueue.pop();
//                hasFrame = true;
//            } else {
//                hasFrame = false;
//            }
//        }
//
//        if (hasFrame) {
//            InsertKeyFrame(nowFrame);
//        } else {
//            usleep(30000);      // sleep 33 ms
//        }
//    }

}

void Mapping::AddFrameToQ(ImageFrame &f)
{
    std::lock_guard<std::mutex> lock(mQueueMutex);
    keyFrameQueue.push(f);
}

void Mapping::InsertKeyFrame(ImageFrame &f)
{
//    ImageFrame* refFrame = f.mRefFrame;
//
//    // triangulate new points
//    TriangulateNewPoints(*refFrame, f);
//
//    // extract new FAST
//    f.extractFAST();
//
//    // fuse FAST
//    vector<int> ref = f.fuseFAST();
//
//    // extract patch discriptors
//    f.extractPatch();
//
//    // set map points
//    int num_mappoints = 0;
//    for (int i = 0, _end = (int)ref.size(); i < _end; i++) {
//        if (ref[i] > 0) {
//            Map_2d_3d::left_const_iterator iter = 
//                    refFrame->map_2d_3d.left.find(ref[i]);
//            if (iter != refFrame->map_2d_3d.left.end()) {
//                f.map_2d_3d.insert(
//                        Map_2d_3d_key_val(i, iter->second));
//                num_mappoints++;
//            }
//        }
//    }
//
//    // set keyFrame
//    f.isKeyFrame = true;
//
//    // make new keyframe
//    ImageFrame * nkf = new ImageFrame(f);
//    nkf->trackedPoints = nkf->points;
//    nkf->undisTrackedPoints = nkf->undisPoints;
//
//    // insert keyFrame
//    {
//        std::lock_guard<std::mutex> lock(mMapMutex);
//        keyFrames.push_back(nkf);
//    }
//
//    // update tracker's refFrame
//    tracker->updateRefFrame(nkf);

}



void Mapping::TriangulateNewPoints(ImageFrame& lf, ImageFrame& rf)
{
//    cv::Mat T1, T2, pts_4d;
//    cv::hconcat(lf.R, lf.t, T1);
//    cv::hconcat(rf.R, rf.t, T2);
//
//    vector< cv::Point2f > pt_1, pt_2;
//    vector< int > pt_idx;
//    for (int i = 0, _end = (int)lf.points.size(); i < _end; ++i) {
//        if (rf.undisTrackedPoints[i].x > 0) {          // first should has crospondence
//            Map_2d_3d::left_const_iterator iter = 
//                    lf.map_2d_3d.left.find(i);
//            if (iter != lf.map_2d_3d.left.end()) {   // second should has 3d map point
//                
//            } else {                                   // if no 3d point, add 2d-2d pair
//                pt_idx.push_back(i);
//                pt_1.push_back(K->pixel2device(
//                            lf.undisPoints[i].x, 
//                            lf.undisPoints[i].y));
//                pt_2.push_back(K->pixel2device(
//                            rf.undisTrackedPoints[i].x, 
//                            rf.undisTrackedPoints[i].y));
//            }
//        }
//    }
//
//    if ( (int)pt_1.size() == 0 ) {
//        printf("No new points to triangulate!\n");
//        return;
//    } 
//
//    cout << "Triangulate new map point:" << endl;
//    cv::triangulatePoints(T1, T2, pt_1, pt_2, pts_4d);
//
//    // get relative R, t
//    cv::Mat R = rf.R * lf.R.t();
//    cv::Mat t = -rf.R*lf.R.t()*lf.t + rf.t;
//
//    bool checkPoints = CheckPoints(R, t, pts_4d);
//
//    if (!checkPoints) {
//        printf("Triangulate inliers num too small!\n");
//    }
//
//    {
//        std::lock_guard<std::mutex> lock(mMapMutex);
//        for (int i = 0; i < pts_4d.cols; i++) {
//            float w = pts_4d.at<float>(3, i);
//            if (w!=0) {
//                cv::Point3f* mpt = new cv::Point3f(
//                        pts_4d.at<float>(0, i)/w,
//                        pts_4d.at<float>(1, i)/w,
//                        pts_4d.at<float>(2, i)/w);
//
//                cout << *mpt << endl;
//
//                mapPoints.insert(mpt);     // Insert map point pointer to std::set
//                lf.map_2d_3d.insert(  // Insert bimap key-val to boost::bimap in lf
//                        Map_2d_3d_key_val(pt_idx[i], mpt));
//            }
//        }
//    }

}

bool Mapping::CheckPoints(cv::Mat &R, cv::Mat &t, cv::Mat &pts)
{
    //cout << endl << "CheckPoints:" << endl;

    //cout << "R:" << endl;
    //cout << R << endl;

    //cout << "t:" << endl;
    //cout << t << endl;

    int inliers = 0;
    // check parallax
    cv::Mat O1O2 = -R.t()*t; 
    for (int i = 0; i < pts.cols; i++) {
        cv::Mat pts_3d(3, 1, CV_64FC1);
        float w = pts.at<float>(3, i);
        pts_3d.at<double>(0) = pts.at<float>(0, i) / w;
        pts_3d.at<double>(1) = pts.at<float>(1, i) / w;
        pts_3d.at<double>(2) = pts.at<float>(2, i) / w;

        if (pts_3d.at<double>(2)<0) {
            pts.at<float>(3, i) = 0;   // mark as outlier
            continue;
        }

        cv::Mat O2P = pts_3d - O1O2;

        //cout << "O1P:" << Converter::getImageType(pts_3d.type()) << endl;
        //cout << pts_3d << endl;

        //cout << "O2P:" << Converter::getImageType(O2P.type()) << endl;
        //cout << O2P << endl;

        double cosAngle = pts_3d.dot(O2P) / (cv::norm(pts_3d) * cv::norm(O2P));

        if (cosAngle > 0.999) {
            pts.at<float>(3, i) = 0;   // mark as outlier
            //printf("Point disparty too smalll , cosAngle: %f\n", cosAngle);
        } else {
            //printf("Point disparty , cosAngle: %f\n", cosAngle);
            inliers++;
        }
    }

    double ratio = (double)inliers/(double)pts.cols;
    if (ratio < 0.5) {
        printf("Triangulation inliers too small !\n");
        return false;
    }

    //printf("Triangulation inliers num: %d ratio: %f\n", inliers, ratio);

    return true;

}

void Mapping::ClearMap()
{
    // delete points
    for (set< Measure3D* >::iterator iter = mapPoints.begin(),
            i_end = mapPoints.end(); iter != i_end; iter++) {
        Measure3D* p = *iter;
        if (p != NULL)
            delete(p);
    }
    mapPoints.clear();

    // delete keyframes
    for (set< ImageFrame* >::iterator iter = keyFrames.begin(),
            i_end = keyFrames.end(); iter != i_end; iter++) {
        ImageFrame* p = *iter;
        if (p != NULL) 
            delete(p);
    }
    keyFrames.clear();
}
