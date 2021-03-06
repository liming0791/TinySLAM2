#include "VisionTracker.h"
#include "MedianFilter.h"
#include "Converter.h"
#include "Timer.h"

void VisionTracker::TrackMonocular(ImageFrame& f)
{
    if (state == NOTINITIALIZED) {
        initializer.SetFirstFrame(&f);
        state = INITIALIZING;
    } else if (state == INITIALIZING) {
        if (initializer.TryInitialize(&f)) {
            {
                std::lock_guard<std::mutex> lock(mRefFrameMutex);
                refFrame = initializer.k2;
            }
            state = INITIALIZED;
        }
    } else {
        ImageFrame* kf;
        {
            std::lock_guard<std::mutex> lock(mRefFrameMutex);
            kf = refFrame;
        }
        f.opticalFlowMeasure(*kf);
        //TrackPose3D2D(*kf, f);
    }
}

void VisionTracker::TrackMonocularLocal(ImageFrame& f)
{
    if (state == NOTINITIALIZED) {
        initializer.SetFirstFrame(&f);
        state = INITIALIZING;
    } else if (state == INITIALIZING) {
        if (initializer.TryInitialize(&f)) {

            AddTrace(*initializer.k1);      // add camera trace
            AddTrace(*initializer.k2);

            {
                std::lock_guard<std::mutex> lock(mRefFrameMutex);
                refFrame = initializer.k2;
            }
            refFrames.insert(refFrame);
            state = INITIALIZED;
        }
    } else {
        ImageFrame* refF;
        {
            std::lock_guard<std::mutex> lock(mRefFrameMutex);
            refF = refFrame;
        }

        cout << endl;

        //TrackByRefFrame(*refF, f);
        TIME_BEGIN();
        TrackByLastFrame(f);
        TIME_END("TrackByLastFrame");

        TIME_BEGIN();
        TrackLocalMap(f);
        TIME_END("TrackLocalMap");


        mR = f.R.clone();
        mt = f.t.clone();

        AddTrace(f);    // add camera trace
    }
}

void VisionTracker::TrackByRefFrame(ImageFrame& ref, ImageFrame& f)
{

    // optical flow from last
    double ratio;
    if (isFirstFellow) {
        ratio = f.opticalFlowMeasure(ref);
        isFirstFellow = false;
    } else {
        ratio = f.opticalFlowMeasure(lastFrame);
    }
    lastFrame = f;
    ratio = ratio / (double)ref.measure2ds.size();
    printf("TrackFeature OpticalFlow ratio: %f\n", ratio);

    //TrackPose3D2D(kf, f);
    // track relative pose 3d-2d
    vector< cv::Point3f > pt3d;
    vector< cv::Point2f > pt2d;

    const double *R_data = ref.R.ptr<double>(0);
    const double *t_data = ref.t.ptr<double>(0);

    for (int i = 0, _end = (int)f.measure2ds.size();
            i < _end; i++) {
        Measure2D* pMea = &(f.measure2ds[i]);
        Measure2D* pRefMea = pMea->ref2d;
        if (pMea->valid && 
                pRefMea->valid &&
                pRefMea->ref3d != NULL) {
            cv::Point3f p = pRefMea->ref3d->pt;
            pt3d.push_back(cv::Point3f(
                        R_data[0]*p.x + R_data[1]*p.y + R_data[2]*p.z + t_data[0],
                        R_data[3]*p.x + R_data[4]*p.y + R_data[5]*p.z + t_data[1],
                        R_data[6]*p.x + R_data[7]*p.y + R_data[8]*p.z + t_data[2]));
            pt2d.push_back(pMea->undisPt);
        }
    }

    cv::Mat R, r, t;

    // solve pnp ransac
    cv::Mat KMat = (cv::Mat_<double>(3, 3) 
            << K->fx, 0, K->cx, 0, K->fy, K->cy, 0, 0, 1);
    TIME_BEGIN();
    cv::solvePnPRansac(pt3d, pt2d, KMat, cv::Mat(), r, t);
    TIME_END("Solve PnP");
    cv::Rodrigues(r,R);
    cout << "solve pnp res:" << endl;
    cout << R << endl;
    cout << t << endl;

    // BA
    vector<int> inliers;
    TIME_BEGIN();
    BA3D2D(pt3d, pt2d, R, t, inliers);
    TIME_END("BA3D2D");
    cout << "BA res:" << endl;
    cout << R << endl;
    cout << t << endl;

    //if (ratio < 0.6 && countFromLastKeyFrame > 10) {
    //    //InsertKeyFrame(kf, f);
    //    map->AddFrameToQ(f);
    //    countFromLastKeyFrame = 0;
    //}
    
    mR = R*ref.R;
    mt = R*ref.t + t;

    f.R = mR.clone();
    f.t = mt.clone();
}

void VisionTracker::TrackByLastFrame(ImageFrame& f)
{

    // optical flow from last
    double ratio;
    if (isFirstFellow) {
        ratio = f.opticalFlowMeasure(*refFrame, -1);
        isFirstFellow = false;
    } else {
        ratio = f.opticalFlowMeasure(lastFrame, -1);
    }
    lastFrame = f;
    //ratio = ratio / (double)refFrame->measure2ds.size();
    printf("TrackFeature OpticalFlow num: %f\n", ratio);

    // track relative pose 3d-2d
    vector< cv::Point3f > pt3d;
    vector< cv::Point2f > pt2d;
    vector< Measure3D* > m3ds;
    pt3d.reserve(f.measure2ds.size());
    pt2d.reserve(f.measure2ds.size());
    m3ds.reserve(f.measure2ds.size());

    for (int i = 0, _end = (int)f.measure2ds.size(); i < _end; i++) {

        Measure2D* pMea = &(f.measure2ds[i]);
        Measure2D* pRefMea = pMea->ref2d;

        if (pMea->valid && pRefMea->valid && pRefMea->ref3d != NULL && pRefMea->ref3d->valid) {
            
            // push back points
            cv::Point3f p = pRefMea->ref3d->pt;
            pt3d.push_back(p);              // 3d points
            pt2d.push_back(pMea->undisPt);  // 2d points
            m3ds.push_back(pRefMea->ref3d);

            // bind measure2d and measure3d
            pMea->ref3d = pRefMea->ref3d;
        }
    }

    cv::Mat R, r, t;

    // solve pnp ransac
    cv::Mat KMat = (cv::Mat_<double>(3, 3) 
            << K->fx, 0, K->cx, 0, K->fy, K->cy, 0, 0, 1);
    TIME_BEGIN();
    cv::solvePnPRansac(pt3d, pt2d, KMat, cv::Mat(), r, t);
    TIME_END("Solve PnP");
    cv::Rodrigues(r,R);
    //cout << "solve pnp res:" << endl;
    //cout << R << endl;
    //cout << t << endl;

    // BA
    vector<int> inliers;
    TIME_BEGIN();
    //BA3D2DOnlyPose(pt3d, pt2d, R, t, inliers);
    BA3D2D(pt3d, pt2d, R, t, inliers);
    TIME_END("BA3D2D");
    //cout << "BA res:" << endl;
    //cout << R << endl;
    //cout << t << endl;
    
    // update 3d points
    //for (int i = 0, _end = (int)m3ds.size(); i < _end; i++ ) {
    //    m3ds[i]->pt = pt3d[i];
    //}

    f.R = R.clone();
    f.t = t.clone();

}

void VisionTracker::BA3D2D(
        vector< cv::Point3f > & points_3d,
        vector< cv::Point2f > & points_2d,
        cv::Mat& R, cv::Mat& t, vector<int> &inliers)
{
    // g2o
    typedef g2o::BlockSolver< g2o::BlockSolverTraits<6,3> > Block;  // pose 6, landmark 3
    Block::LinearSolverType* linearSolver = 
        new g2o::LinearSolverCSparse<Block::PoseMatrixType>(); // linearSolver
    Block* solver_ptr = new Block ( linearSolver );     // blockSolver
    g2o::OptimizationAlgorithmLevenberg* solver = 
        new g2o::OptimizationAlgorithmLevenberg ( solver_ptr );
    g2o::SparseOptimizer optimizer;
    optimizer.setAlgorithm ( solver );

    // Set Frame vertex
    g2o::VertexSE3Expmap* pose = new g2o::VertexSE3Expmap(); // camera pose
    pose->setId ( 0 );
    Eigen::Matrix3d R_mat;
    cv::cv2eigen(R, R_mat);
    pose->setEstimate ( g2o::SE3Quat(R_mat,
                Eigen::Vector3d(t.at<double>(0), t.at<double>(1), t.at<double>(2))) );
    optimizer.addVertex ( pose );

    // Set MapPoint vertices
    int index = 1;
    for ( const cv::Point3f p:points_3d )   // landmarks
    {
        g2o::VertexSBAPointXYZ* point = new g2o::VertexSBAPointXYZ();
        point->setId ( index++ );
        point->setEstimate ( Eigen::Vector3d ( p.x, p.y, p.z ) );
        point->setFixed(true);
        point->setMarginalized ( true ); // g2o set marg 
        optimizer.addVertex ( point );
    }

    // parameter: camera intrinsics
    g2o::CameraParameters* camera = new g2o::CameraParameters (
            (K->fx + K->fy)/2, 
            Eigen::Vector2d( K->cx, K->cy ), 0);
    camera->setId ( 0 );
    optimizer.addParameter ( camera );

    // Set 2d Point edges
    vector< g2o::EdgeProjectXYZ2UV* > edges;
    edges.reserve(points_2d.size());
    index = 1;
    for ( const cv::Point2f p:points_2d )
    {
        g2o::EdgeProjectXYZ2UV* edge = new g2o::EdgeProjectXYZ2UV();
        edge->setId ( index );
        edge->setVertex ( 0, dynamic_cast<g2o::VertexSBAPointXYZ*> ( optimizer.vertex ( index ) ) );
        edge->setVertex ( 1, pose );
        edge->setMeasurement ( Eigen::Vector2d ( p.x, p.y ) );
        edge->setParameterId ( 0,0 );
        edge->setInformation ( Eigen::Matrix2d::Identity() );
        optimizer.addEdge ( edge );
        index++;

        edges.push_back(edge);
    }

    //TIME_BEGIN()
    //optimizer.initializeOptimization();
    //optimizer.optimize ( 20 );
    ////TIME_END("g2o optimization")
    
    inliers.resize(points_3d.size());
    std::fill(inliers.begin(), inliers.end(), 1);

    double chi2Thres = 3;
    int nBad = 0;
    for (int it = 0; it < 2; it++) {

        cout << "Iter: " << it << endl;

        pose->setEstimate ( g2o::SE3Quat(R_mat,
                Eigen::Vector3d(t.at<double>(0), t.at<double>(1), t.at<double>(2))) );

        //TIME_BEGIN()
        optimizer.initializeOptimization(0);
        optimizer.optimize ( 10 );
        //TIME_END("g2o optimization")
        
        cout << "edge size: " << optimizer.edges().size() << endl;
        
        nBad = 0;
        for (int i = 0, _end = (int)edges.size(); i < _end; i++) {

            g2o::EdgeProjectXYZ2UV* e = edges[i];
            
            if (inliers[i] == 0) {
                e->computeError();
            }

            if (e->chi2() > chi2Thres) {
                inliers[i] = 0;
                e->setLevel(1);
                nBad++;
            } else {
                inliers[i] = 1;
                e->setLevel(0);
            }
        }
        cout << "nBad: " << nBad << endl;

        if (optimizer.edges().size() < 10)
            break;
    }

    //cout<<endl<<"after optimization:"<<endl;
    //cout<<"T="<<endl<<Eigen::Isometry3d ( pose->estimate() ).matrix() <<endl;

    // set optimization result back, do not forget this step
    Eigen::Matrix4d T = Eigen::Isometry3d ( pose->estimate() ).matrix();
    cv::Mat res;
    cv::eigen2cv(T, res);
    R = res(cv::Rect(0,0,3,3)).clone();
    t = res(cv::Rect(3,0,1,3)).clone();

    //// log some information
    //cout << "pints 3d before BA:" << endl;
    //for (const cv::Point3f p:points_3d) {
    //    cout << p << endl;
    //} 

    //cout << "points 3d after BA:" << endl;
    //for (int i = 0; i < points_3d.size(); i++ ) {
    //    g2o::VertexSBAPointXYZ* vertex = dynamic_cast< g2o::VertexSBAPointXYZ* >
    //            ( optimizer.vertex(1+i) );
    //    Eigen::Vector3d p = vertex->estimate();
    //    cout << cv::Point3f(p[0], p[1], p[2]) << endl;
    //    //points_3d[i] = cv::Point3f(p[0], p[1], p[2]);
    //}

    //cout << "edge errors:" << endl;
    //for (int i = 0; i < points_3d.size(); i++) {
    //    g2o::EdgeProjectXYZ2UV* e = edges[i];
    //    e->computeError();
    //    cout << e->chi2() << endl;
    //}

}

void VisionTracker::BA3D2DOnlyPose(
        vector< cv::Point3f > & points_3d,
        vector< cv::Point2f > & points_2d,
        cv::Mat& R, cv::Mat& t, vector<int> &inliers)
{
    // g2o
    typedef g2o::BlockSolver< g2o::BlockSolverTraits<6,3> > Block;  // pose 6, landmark 3
    Block::LinearSolverType* linearSolver = 
        new g2o::LinearSolverCSparse<Block::PoseMatrixType>(); // linearSolver
    Block* solver_ptr = new Block ( linearSolver );     // blockSolver
    g2o::OptimizationAlgorithmLevenberg* solver = 
        new g2o::OptimizationAlgorithmLevenberg ( solver_ptr );
    g2o::SparseOptimizer optimizer;
    optimizer.setAlgorithm ( solver );

    // Set Frame vertex
    g2o::VertexSE3Expmap* pose = new g2o::VertexSE3Expmap(); // camera pose
    pose->setId ( 0 );
    Eigen::Matrix3d R_mat;
    cv::cv2eigen(R, R_mat);
    pose->setEstimate ( g2o::SE3Quat(R_mat,
                Eigen::Vector3d(t.at<double>(0), t.at<double>(1), t.at<double>(2))) );
    pose->setFixed(false);
    optimizer.addVertex ( pose );

    // Set 2d Point edges
    vector< g2o::EdgeSE3ProjectXYZOnlyPose* > edges;
    edges.reserve(points_2d.size());
    for ( int i = 0, _end = (int)points_2d.size(); i < _end; i++ )
    {
        g2o::EdgeSE3ProjectXYZOnlyPose* edge = new g2o::EdgeSE3ProjectXYZOnlyPose();
        edge->setVertex ( 0, pose );
        edge->setMeasurement ( Eigen::Vector2d ( points_2d[i].x, points_2d[i].y ) );
        edge->setInformation ( Eigen::Matrix2d::Identity() );

        g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
        rk->setDelta(2.447);
        edge->setRobustKernel(rk);

        edge->fx = K->fx;
        edge->fy = K->fy;
        edge->cx = K->cx;
        edge->cy = K->cy;

        edge->Xw[0] = points_3d[i].x;
        edge->Xw[1] = points_3d[i].y;
        edge->Xw[2] = points_3d[i].z;

        optimizer.addEdge ( edge );

        edges.push_back(edge);
    }

    inliers.resize(points_2d.size());
    std::fill(inliers.begin(), inliers.end(), 1);

    const float chi2Mono[4]={5.991,5.991,5.991,5.991};
    const int its[4]={10,10,10,10};

    cout << "BA points size: " << points_2d.size() << endl;

    int nBad = 0;
    for (int it = 0; it < 2; it++) {

        cout << "iter: " << it << endl;

        pose->setEstimate ( g2o::SE3Quat(R_mat,
                Eigen::Vector3d(t.at<double>(0), t.at<double>(1), t.at<double>(2))) );

        //TIME_BEGIN()
        optimizer.initializeOptimization(0);
        optimizer.optimize ( its[it] );
        //TIME_END("g2o optimization")
        
        cout << "edge size: " << optimizer.edges().size() << endl;
        
        nBad = 0;
        for (int i = 0, _end = (int)edges.size(); i < _end; i++) {

            g2o::EdgeSE3ProjectXYZOnlyPose* e = edges[i];
            
            if (inliers[i] == 0) {
                e->computeError();
            }

            if (e->chi2() > chi2Mono[it]) {
                inliers[i] = 0;
                e->setLevel(1);
                nBad++;
            } else {
                inliers[i] = 1;
                e->setLevel(0);
            }

            if (it==2) {
                e->setRobustKernel(0);
            }

        }

        cout << "nBad: " << nBad << endl;

        if (optimizer.edges().size() < 10)
            break;
    }

    // set optimization result back, do not forget this step
    Eigen::Matrix4d T = Eigen::Isometry3d ( pose->estimate() ).matrix();
    cv::Mat res;
    cv::eigen2cv(T, res);
    R = res(cv::Rect(0,0,3,3)).clone();
    t = res(cv::Rect(3,0,1,3)).clone();

}

// project and wrap to suitable level, search based on extracted FAST 
void VisionTracker::TrackLocalMap(ImageFrame& f)
{

    // result to show, only for dev
    vector< cv::Point2f > pt1, pt2;
    vector< int > l1, l2;
    vector< cv::Point2f > pflow, pProj;
    vector< Measure3D* > p3d;
    vector< Measure2D* > ptrTrackedM2d;

    pt1.reserve(map->mapPoints.size());
    pt2.reserve(map->mapPoints.size());
    l1.reserve(map->mapPoints.size());
    l2.reserve(map->mapPoints.size());
    pflow.reserve(map->mapPoints.size());
    pProj.reserve(map->mapPoints.size());
    p3d.reserve(map->mapPoints.size());
    ptrTrackedM2d.reserve(map->mapPoints.size());
    // end

    // extract FAST
    TIME_BEGIN();
    f.extractFASTGrid();
    TIME_END("extractFAST");

    // project each tracked map point to image frame f
    double *_R = f.R.ptr<double>(0);
    double *_t = f.t.ptr<double>(0);
    cv::Mat A = (K->CamK)*f.R*(K->CamKinv);

    ImageFrame* pF;
    for (int i = 0, _end = (int)f.measure2ds.size(); i < _end; i++) {

        if (f.measure2ds[i].ref3d == NULL)
            continue;

        Measure3D* pM3d = f.measure2ds[i].ref3d;
        Measure2D* pM2d = pM3d->ref2d;
        pF = pM2d->refFrame;

        //cout << endl << " For 3d point: " << pM3d->pt << endl;
        //cout << "Ref 2d point: " << pM2d->pt << endl;

        // for dev
        p3d.push_back(pM3d);
        pt1.push_back(pM2d->pt);
        l1.push_back(pM2d->levelIdx);

        double X = _R[0]*pM3d->pt.x+_R[1]*pM3d->pt.y+_R[2]*pM3d->pt.z+_t[0];
        double Y = _R[3]*pM3d->pt.x+_R[4]*pM3d->pt.y+_R[5]*pM3d->pt.z+_t[1];
        double Z = _R[6]*pM3d->pt.x+_R[7]*pM3d->pt.y+_R[8]*pM3d->pt.z+_t[2];

        cv::Point2f pt2d = K->Proj3Dto2D(X, Y, Z);

        // for dev
        //pProj.push_back(pt2d);
        // end
        //cout << "Project 2d point: " << pt2d << endl;
        // TODO::distort projected pt2d
        
        // compute wrap
        cv::Mat A_ = A * pM3d->pt.z / Z;

        //cv::Mat B_ = B / Z;
        //cv::Mat uv_ori(3, 1, CV_64FC1);
        //uv_ori.at<double>(0) = pM2d->undisPt.x;
        //uv_ori.at<double>(1) = pM2d->undisPt.y;
        //uv_ori.at<double>(2) = 1;
        //cv::Mat uv_new = A_ * uv_ori + B_;
        //cout << endl << "uv_new wrap: " << uv_new << endl;
        //cout << "uv_new proj: " << pt2d << endl;
        //cout << "A_: " << A_ << endl;

        cv::Mat wrapM = A_(cv::Rect(0,0,2,2)).clone();
        double* M_ = wrapM.ptr<double>(0);

        //cout << "wrapM: " << wrapM << endl;
        double area = M_[0] * M_[3] - M_[1] * M_[2];
        //cout << "area: " << area << endl;

        int searchLevel = pM2d->levelIdx;
        //cout << "ori level: " << searchLevel << endl;

        while (area > 3 && searchLevel < 3) {
            searchLevel++;
            area *= 0.25;
        }

        if (area > 3 || area < 0.25) {
            // TODO::too near or too far
        }
        //printf("search level: %d\n", searchLevel);
        
        // search around pt2d
        bool matchFinded = false;
        double sec_best_score = 999999.;
        int sec_best_x, sec_best_y;

        double best_score = 999999.;
        int best_x, best_y;

        double scaleFactor = f.levels[searchLevel].scaleFactor;
        //cout << "ScaleFactor: " << scaleFactor << endl;

        double distThre = 49;
        //cout << "dist threshold: " << distThre << endl;

        for (cv::Point2f pt:f.levels[searchLevel].points) {

            //cout << "ori level fast point: " << pt << endl;

            double sx = pt.x * scaleFactor;
            double sy = pt.y * scaleFactor;

            //cout << "FAST position: " << sx << " " << sy << endl;
            //cout << "Project position: " << pt2d.x << " " << pt2d.y << endl;

            double dist = (pt2d.x-sx)*(pt2d.x-sx) + (pt2d.y-sy)*(pt2d.y-sy);

            if ( dist >  distThre) {
                continue;
            }

            cv::Point2f sp(sx, sy);
            double score = SSD(pF->levels[0].image, pM2d->pt, pM2d->levelIdx,
                    f.levels[0].image, sp, searchLevel, M_);
            //cout << "similarity: " << score << endl;

            if (score > 10000)
                continue;

            if (score < best_score) {
                best_score = score;
                best_x = sx;
                best_y = sy;
                matchFinded = true;
            } else if (score < sec_best_score) {
                sec_best_score = score;
                sec_best_x = sx;
                sec_best_y = sy;
            }

        }

        // for dev
        if (matchFinded) {
            pt2.push_back(cv::Point2f(best_x, best_y));
            f.measure2ds[i].pt = cv::Point2f(best_x, best_y);
            f.measure2ds[i].undisPt = K->undistort(best_x, best_y);
        }
        else {
            pt2.push_back(cv::Point2f(-1, -1));
        }
        l2.push_back(searchLevel);
        // end
        
    }

    // refine pose
    // track relative pose 3d-2d
    vector< cv::Point3f > pt3d;
    vector< cv::Point2f > pt2d;
    vector< int > idxs;
    pt3d.reserve(p3d.size());
    pt2d.reserve(p3d.size());
    idxs.reserve(p3d.size());

    for (int i = 0, _end = (int)pt1.size(); i < _end; i++) {

        if (pt2[i].x < 0)
            continue;

        if (!p3d[i]->valid)
            continue;

        // push back points
        pt3d.push_back(p3d[i]->pt);              // 3d points
        pt2d.push_back(pt2[i]);  // 2d points
        idxs.push_back(i);
    }

    // BA
    cout << "refine points num: " << pt3d.size() << endl;
    vector<int> inliers;
    TIME_BEGIN();
    //BA3D2DOnlyPose(pt3d, pt2d, f.R, f.t, inliers);
    BA3D2D(pt3d, pt2d, f.R, f.t, inliers);
    TIME_END("fefine BA3D2D");
    //cout << "fefine BA res:" << endl;
    //cout << f.R << endl;
    //cout << f.t << endl;
    
    for (int i = 0, _end = (int)inliers.size(); i < _end; i++) {
        if (inliers[i]==0) {
            int idx = idxs[i];
            p3d[idx]->outlierNum++;
            if (p3d[idx]->outlierNum > 3) {
                p3d[idx]->valid = false;
            }
        }
    }

    // draw result, for dev
    cv::Mat res(480, 640*2, CV_8UC3);
    cv::Mat region1(res, cv::Rect(0, 0, 640, 480));
    cv::Mat region2(res, cv::Rect(640, 0, 640, 480));
    cv::cvtColor(pF->levels[0].image, region1, CV_GRAY2BGR);
    cv::cvtColor(f.levels[0].image, region2, CV_GRAY2BGR);
    cv::Scalar color(0, 255, 0);
    for (int i = 0, _end = (int)pt1.size(); i < _end; i++) {

        if (pt2[i].x < 0)
            continue;

        // draw match point
        cv::Scalar rcolor(rand()%255,rand()%255,rand()%255);
        cv::circle(res, pt1[i], 3.5*(1<<l1[i]), 
                rcolor);
        cv::circle(res, cv::Point2f(pt2[i].x+640, pt2[i].y),
                3.5*(1<<l2[i]), 
                rcolor);
        cv::line(res, pt1[i], 
                cv::Point2f(pt2[i].x+640, pt2[i].y),
                rcolor);

        //if (pflow[i].x < 0)
        //    continue;

        //if (cv::norm(pflow[i] - pt2[i]) < 3)
        //    continue;

        //// draw flow point
        //cv::circle(res, cv::Point2f(pflow[i].x+640, pflow[i].y),
        //        3.5*(1<<l1[i]), cv::Scalar(0,0,255));
        //cv::line(res, cv::Point2f(pflow[i].x+640, pflow[i].y), 
        //        cv::Point2f(pt2[i].x+640, pt2[i].y),
        //        cv::Scalar(0,0,255));

        // draw proj point
        //cv::circle(res, cv::Point2f(pProj[i].x+640, pProj[i].y),
        //        3.5*(1<<l1[i]), cv::Scalar(255,0,0));
        //cv::line(res, cv::Point2f(pProj[i].x+640, pProj[i].y), 
        //        cv::Point2f(pt2[i].x+640, pt2[i].y),
        //        cv::Scalar(255, 0, 0));

    }

    cv::imshow("match", res);
    //cv::waitKey(-1);
    // end

}

//void VisionTracker::TrackMonocularNewKeyFrame(ImageFrame& f)
//{
//    if (state == NOTINITIALIZED) {
//        initializer.SetFirstFrame(&f);
//        state = INITIALIZING;
//    } else if (state == INITIALIZING) {
//        if (initializer.TryInitializeByG2O(&f)) {
//            {
//                std::lock_guard<std::mutex> lock(mRefFrameMutex);
//                refFrame = initializer.resFirstFrame;
//                lastFrame = initializer.lastFrame;
//            }
//            state = INITIALIZED;
//        }
//    } else {
//        ImageFrame* kf;
//        {
//            std::lock_guard<std::mutex> lock(mRefFrameMutex);
//            kf = refFrame;
//        }
//        TrackByKeyFrame(*kf, f);
//    }
//}
//
//void VisionTracker::TryInitialize(ImageFrame& f)
//{
//
//    // Track FAST Points
//    //int num_tracked = f.opticalFlowFAST(*refFrame);
//    // check trakced points ratio
//    //double ratio_tracked = 
//    //        num_tracked / (double)f.undisTrackedPoints.size();
//
//    double ratio_tracked = TrackFeatureOpticalFlow(*refFrame, f);
//
//    if (ratio_tracked < 0.5) {
//        printf("Initialize Tracked points num too small!"
//                " less than 0.3\n");
//        return;
//    }
//
//    // Track Pose 2D-2D
//    // prepare tracked points
//    std::vector< cv::Point2f > lp, rp;
//    std::vector< int > pt_idx;
//    lp.reserve(f.undisTrackedPoints.size());
//    rp.reserve(f.undisTrackedPoints.size());
//    pt_idx.reserve(f.undisTrackedPoints.size());
//    for (int i = 0, _end = (int)f.undisTrackedPoints.size(); 
//            i < _end; i++ ) {
//        if (f.undisTrackedPoints[i].x > 0) {
//            lp.push_back(refFrame->undisPoints[i]);
//            rp.push_back(f.undisTrackedPoints[i]);
//            pt_idx.push_back(i);
//        }
//    }
//
//    // check disparty
//    double disparty = 0;
//    for (int i = 0, _end = (int)lp.size(); i < _end; i++) {
//        disparty = disparty + (lp[i].x - rp[i].x)*(lp[i].x - rp[i].x)
//                + (lp[i].y - rp[i].y)*(lp[i].y - rp[i].y);
//    }
//    disparty = sqrt(disparty/(double)lp.size()) ;
//    if ( disparty < K->width/32.0 ) {
//        printf("Initialize disparty too small, less than %f average!\n", K->width/32.0);
//        return;
//    }
//
//    // find essentialmat
//    cv::Mat inliers;
//    cv::Mat essential_matrix = cv::findEssentialMat(
//            lp, rp, 
//            (K->fx + K->fy)/2, cv::Point2d(K->cx, K->cy), 
//            cv::RANSAC, 0.999999, 3.0, inliers);
//    int num_inliers = 0;
//    for (int i = 0, _end = (int)lp.size(); i < _end; i++) {
//        if (inliers.at<unsigned char>(i) == 1) {
//            ++num_inliers;
//        } else {
//            int idx = pt_idx[i];
//            f.trackedPoints[idx].x = f.trackedPoints[idx].y = 0;
//            f.undisTrackedPoints[idx].x = f.undisTrackedPoints[idx].y = 0;
//        }
//    }
//    double ratio_inliers = (double)num_inliers / (int)lp.size();
//    if (ratio_inliers < 0.9) {
//       printf("Initialize essential matrix inliers num too small!"
//               " less than 0.9\n"); 
//       return;
//    }
//    cout << "essential_matrix: " << endl
//        << essential_matrix << endl;
//
//    // recovery pose
//    cv::Mat R, t;
//    cv::recoverPose(essential_matrix, lp, rp, R, t, 
//            (K->fx + K->fy)/2, cv::Point2d(K->cx, K->cy), inliers);
//    int pose_num_inliers = 0;
//    for (int i = 0, _end = (int)lp.size(); i < _end; i++) {
//        if (inliers.at<unsigned char>(i) == 1) {
//            ++pose_num_inliers;
//        } else {
//            int idx = pt_idx[i];
//            f.trackedPoints[idx].x = f.trackedPoints[idx].y = 0;
//            f.undisTrackedPoints[idx].x = f.undisTrackedPoints[idx].y = 0;
//        }
//    }
//    double ratio_pose_inliers = (double)pose_num_inliers / (int)lp.size();
//    if (ratio_pose_inliers < 0.9) {
//        printf("Initialize recover pose inliers num %f too small!"
//                " less than 0.9\n", ratio_pose_inliers); 
//        return;
//    }
//
//    // check shift
//    double shift = sqrt( t.at<double>(0)*t.at<double>(0) +
//            + t.at<double>(1)*t.at<double>(1) +
//            + t.at<double>(2)*t.at<double>(2) );
//
//    if (shift < 0.05) {
//        printf("Initialize recover pose shift too small, less than 0.05!"
//                " Is %f\n", shift);
//        return ;
//    }
//
//    cout << "R: " << endl
//        << R << endl;
//    cout << "t: " << endl
//        << t << endl;
//
//    //cv::Mat Rt = R.t();
//    //cv::Mat _t = -Rt*t;
//
//    mR = R.clone();
//    mt = t.clone();
//
//    f.R = R.clone();
//    f.t = t.clone();
//
//    // Init Mapping
//    map->InitMap(*refFrame, f);
//
//    // Set state
//    state = INITIALIZED;
//}
//
//void VisionTracker::TryInitializeByG2O(ImageFrame& f)
//{
//
//    // Track FAST Points
//    f.opticalFlowFAST(*refFrame);
//
//    // Track Pose 2D-2D
//    std::vector< cv::Point2f > lp, rp;
//    vector< int > pt_idx;
//    lp.reserve(f.undisTrackedPoints.size());
//    rp.reserve(f.undisTrackedPoints.size());
//    pt_idx.reserve(f.undisTrackedPoints.size());
//
//    for (int i = 0, _end = (int)f.undisTrackedPoints.size(); 
//            i < _end; i++ ) {
//        if (f.undisTrackedPoints[i].x > 0) {
//            lp.push_back(refFrame->undisPoints[i]);
//            rp.push_back(f.undisTrackedPoints[i]);
//            pt_idx.push_back(i);
//        }
//    }
//
//    // check tracked percent
//    double ratio_tracked = (double)lp.size() / 
//            (double)f.undisTrackedPoints.size();
//    if (ratio_tracked < 0.3) {
//        printf("Initialize Tracked points num too small!"
//                " less than 0.3\n");
//        return;
//    }
//
//    // check disparty
//    double disparty = 0;
//    for (int i = 0, _end = (int)lp.size(); i < _end; i++) {
//        disparty = disparty + (lp[i].x - rp[i].x)*(lp[i].x - rp[i].x)
//                + (lp[i].y - rp[i].y)*(lp[i].y - rp[i].y);
//    }
//    if ( disparty < 900.0 * (int)lp.size() ) {
//        printf("Initialize disparty too small, less than 30 average!\n");
//        return;
//    }
//    
//    // init by g2o
//    cout << "Set optimizer." << endl;
//
//    // Optimizer
//    g2o::SparseOptimizer optimizer;
//    
//    // linear solver
//    g2o::BlockSolver_6_3::LinearSolverType* linearSolver = new g2o::LinearSolverCSparse< g2o::BlockSolver_6_3::PoseMatrixType >();
//
//    // block solver
//    g2o::BlockSolver_6_3* block_solver = new g2o::BlockSolver_6_3( linearSolver );
//
//    // optimization algorithm
//    g2o::OptimizationAlgorithmLevenberg* algorithm = new g2o::OptimizationAlgorithmLevenberg( block_solver );
//
//    optimizer.setAlgorithm( algorithm );
//    optimizer.setVerbose( false );
//
//    cout << "add pose vertices." << endl;
//    // add pose vertices
//    g2o::VertexSE3Expmap* v1 = new g2o::VertexSE3Expmap();
//    v1->setId(0);
//    v1->setFixed(true);
//    v1->setEstimate(g2o::SE3Quat());
//    optimizer.addVertex(v1);
//
//    g2o::VertexSE3Expmap* v2 = new g2o::VertexSE3Expmap();
//    v2->setId(1);
//    v2->setEstimate(g2o::SE3Quat());
//    optimizer.addVertex(v2);
//
//    cout << "add 3d points vertices" << endl;
//    // add 3d point vertices
//    for (int i = 0, _end = (int)lp.size(); i < _end; i++) {
//        g2o::VertexSBAPointXYZ* v = new g2o::VertexSBAPointXYZ();
//        v->setId(2+i);
//        double z = 10;
//        double x = ( lp[i].x - K->cx ) / K->fx * z;
//        double y = ( lp[i].y - K->cy ) / K->fy * z;
//        v->setMarginalized(true);
//        v->setEstimate( Eigen::Vector3d(x, y, z) );
//        optimizer.addVertex( v );
//    }
//
//    cout << "add camera parameters" << endl;
//    // prepare camera parameters
//    g2o::CameraParameters* camera = new g2o::CameraParameters( (K->fx + K->fy)/2, Eigen::Vector2d(K->cx, K->cy), 0 );
//    camera->setId(0);
//    optimizer.addParameter(camera);
//
//    cout << "add edges" << endl;
//    // prepare edges
//    vector< g2o::EdgeProjectXYZ2UV* > edges;
//    for (int i = 0, _end = (int)lp.size(); i < _end; i++) {
//        g2o::EdgeProjectXYZ2UV* edge = new g2o::EdgeProjectXYZ2UV();
//        edge->setVertex( 0, dynamic_cast< g2o::VertexSBAPointXYZ* > (optimizer.vertex(i+2)) );
//        edge->setVertex( 1, dynamic_cast< g2o::VertexSE3Expmap* > (optimizer.vertex(0)) );
//
//        edge->setMeasurement( Eigen::Vector2d(lp[i].x, lp[i].y) );
//        edge->setInformation( Eigen::Matrix2d::Identity() );
//        edge->setParameterId(0, 0);
//
//        edge->setRobustKernel( new g2o::RobustKernelHuber() );
//        optimizer.addEdge( edge );
//        edges.push_back( edge );
//    }
//
//    for (int i = 0, _end = (int)rp.size(); i < _end; i++) {
//        g2o::EdgeProjectXYZ2UV* edge = new g2o::EdgeProjectXYZ2UV();
//        edge->setVertex( 0, dynamic_cast< g2o::VertexSBAPointXYZ* > (optimizer.vertex(i+2)) );
//        edge->setVertex( 1, dynamic_cast< g2o::VertexSE3Expmap* > (optimizer.vertex(1)) );
//
//        edge->setMeasurement( Eigen::Vector2d(rp[i].x, rp[i].y) );
//        edge->setInformation( Eigen::Matrix2d::Identity() );
//        edge->setParameterId(0, 0);
//
//        edge->setRobustKernel( new g2o::RobustKernelHuber() );
//        optimizer.addEdge( edge );
//        edges.push_back( edge );
//    }
//
//    cout << "optimization" << endl;
//    // optimization
//    optimizer.setVerbose(true);
//    optimizer.initializeOptimization();
//    optimizer.optimize(100);
//
//    // num of inliers
//    vector< int > inliers(edges.size()/2, 1);
//    int num_vert = (int)inliers.size();
//    for ( int i = 0, _end = (int)edges.size(); i < _end; i++) {
//        g2o::EdgeProjectXYZ2UV* e = edges[i];
//        e->computeError();
//        if (e->chi2() > 1) {
//            inliers[i%num_vert] = 0;
//            cout << "error = " << e->chi2() << endl;
//        } 
//    }
//    int num_inliers = 0;
//    for ( int i = 0, _end = (int)inliers.size(); i < _end; i++) {
//        num_inliers += inliers[i];
//    }
//    cout << "num inliers: " << num_inliers << endl;
//
//    // check inliers
//    double ratio_inlier = (double)num_inliers / (double)lp.size();
//    if (ratio_inlier < 0.8) {
//        printf("Inliers too small, less than 0.8 !\n");
//        return;
//    }
//
//    // SE3 estimate
//    g2o::VertexSE3Expmap* v = 
//            dynamic_cast< g2o::VertexSE3Expmap* >( optimizer.vertex(1) );
//    Eigen::Isometry3d pose = v->estimate();
//    // set optimization pose result, do not forget this step
//    Eigen::Matrix4d T = pose.matrix();
//    cv::Mat res;
//    cv::eigen2cv(T, res);
//    cv::Mat R = res.rowRange(0,3).colRange(0,3);
//    cv::Mat t = res.rowRange(0,3).col(3);
//
//    cv::Mat Rt = R.t();
//    cv::Mat _t = -Rt*t;
//
//    mR = Rt.clone();
//    mt = _t.clone();
//
//    f.R = mR.clone();
//    f.t = mt.clone();
//
//    cout << "mR: " << endl;
//    cout << mR <<  endl;
//
//    cout << "mt: " << endl;
//    cout << mt << endl;
//
//    // points estimate
//    for (int i = 0, _end = (int)lp.size(); i < _end; i++ ) {
//        if (inliers[i] == 0)
//            continue;
//        g2o::VertexSBAPointXYZ* v = 
//                dynamic_cast< g2o::VertexSBAPointXYZ* > 
//                ( optimizer.vertex(i+2) );
//        Eigen::Vector3d pos = v->estimate();
//        // set Mapping points and lf points
//        cv::Point3f* mpt = new cv::Point3f(pos[0], pos[1], pos[2]);
//        map->mapPoints.insert(mpt);
//        refFrame->map_2d_3d.insert(
//                Map_2d_3d_key_val(pt_idx[i], mpt));
//    }
//
//    // add new keyFrame, malloc at heap
//    ImageFrame * nkf = new ImageFrame(*refFrame);
//    nkf->isKeyFrame = true;
//    map->keyFrames.push_back(nkf);
//
//    // Set state
//    state = INITIALIZED;
//
//    // update refFrame
//    refFrame = nkf;
//
//}

void VisionTracker::reset()
{
    map->ClearMap();
    state = NOTINITIALIZED;
}

//void VisionTracker::TrackByKeyFrame(ImageFrame& kf, ImageFrame& f)
//{
//
//    countFromLastKeyFrame++;
//    //int numTracked = f.opticalFlowFAST(kf);
//    //double ratio = (double)numTracked / (double)kf.points.size();
//    double ratio = TrackFeatureOpticalFlow(kf, f);
//
//    printf("TrackFeature OpticalFlow ratio: %f\n", ratio);
//
//    TrackPose3D2D(kf, f);
//
//    double dist = cv::norm(-f.R*kf.R.t()*kf.t + f.t, cv::NORM_L2);
//    cout << "Frame dist to keyframe: " << dist << endl;
//
//    if (ratio < 0.6 && countFromLastKeyFrame > 10) {
//        //InsertKeyFrame(kf, f);
//        map->AddFrameToQ(f);
//        countFromLastKeyFrame = 0;
//    }
//}

//void VisionTracker::InsertKeyFrame(ImageFrame& kf, ImageFrame& f)
//{
//    // track by last keyframe
//    //TrackPose3D2DHybrid(kf, f);
//
//    // track by last keyframe, and triangulate new points
//    //TrackPose3D2D(kf, f);
//    TriangulateNewPoints(kf, f);
//
//    // extrack new FAST
//    f.extractFAST();
//
//    // fuse FAST and set mappoints
//    vector<int> ref = f.fuseFAST();
//
//    // extract patch discreptors
//    f.extractPatch();
//
//    // set map points
//    int num_mappoints = 0;
//    for (int i = 0, _end = (int)ref.size(); i < _end; i++) {
//        if (ref[i] > 0) {
//            Map_2d_3d::left_const_iterator iter = kf.map_2d_3d.left.find(ref[i]);
//            if (iter != kf.map_2d_3d.left.end()) {
//                f.map_2d_3d.insert(Map_2d_3d_key_val(i, iter->second));
//                num_mappoints++;
//            }
//        }
//    }
//    printf("set map points num: %d \n", num_mappoints);
//
//    // keyFrame 
//    f.isKeyFrame = true;
//
//    // make new keyframe
//    ImageFrame * nkf = new ImageFrame(f);
//    nkf->trackedPoints = nkf->points;
//    nkf->undisTrackedPoints = nkf->undisPoints;
//    map->keyFrames.push_back(nkf);
//
//    // update refFrame
//    refFrame = nkf;
//
//    // update lastFrame
//    lastFrame = *refFrame;
//
//    // log keyframe size
//    printf("There are %d keyframes\n", (int)map->keyFrames.size());
//}
//
//void VisionTracker::TrackPose2D2D(const ImageFrame& lf, ImageFrame& rf)
//{
//    std::vector< cv::Point2f > lp, rp;
//    lp.reserve(rf.undisTrackedPoints.size());
//    rp.reserve(rf.undisTrackedPoints.size());
//
//    for (int i = 0, _end = (int)rf.undisTrackedPoints.size(); i < _end; i++ ) {
//        if (rf.undisTrackedPoints[i].x > 0) {
//            lp.push_back(lf.undisPoints[i]);
//            rp.push_back(rf.undisTrackedPoints[i]);
//        }
//    }
//
//    cv::Point2d principal_point(K->cx, K->cy);
//    double focal_length = (K->fx + K->fy)/2;
//    cv::Mat essential_matrix = cv::findEssentialMat(lp, rp, focal_length, principal_point);
//    cout << "essential_matrix: " << endl
//        << essential_matrix << endl;
//
//    cv::Mat R, t;
//    cv::recoverPose(essential_matrix, lp, rp, R, t, focal_length, principal_point);
//    cout << "R: " << endl
//        << R << endl;
//    cout << "t: " << endl
//        << t << endl;
//
//    cv::Mat Rt = R.t();
//    cv::Mat _t = -Rt*t;
//
//    mR = Rt.clone();
//    mt = _t.clone();
//
//    rf.R = mR.clone();
//    rf.t = mt.clone();
//
//    TooN::SO3<> Rot;
//    TooN::Vector<3> Trans;
//    Converter::Mat_TooNSO3(mR, Rot);
//    Trans[0] = mt.at<double>(0);
//    Trans[1] = mt.at<double>(1);
//    Trans[2] = mt.at<double>(2);
//    cout << "Rot: " << endl
//        << Rot <<endl;
//    cout << "Trans: " << endl
//        << Trans << endl;
//
//    mTcwNow = rf.mTcw = lf.mTcw * TooN::SE3<>(Rot, Trans);
//    cout << "mTcw: " << endl
//        << rf.mTcw << endl;
//}

//void VisionTracker::TrackPose2D2DG2O(ImageFrame& lf, ImageFrame& rf)
//{
//    // prepare points
//    vector< cv::Point2f > lp, rp;
//    vector< int > pt_idx;
//    lp.reserve(lf.undisPoints.size());
//    rp.reserve(lf.undisPoints.size());
//    pt_idx.reserve(lf.undisPoints.size());
//    for (int i = 0, _end = (int)lf.undisPoints.size(); i < _end; i++) {
//        if (rf.undisTrackedPoints[i].x > 0) {
//            lp.push_back(lf.undisPoints[i]);
//            rp.push_back(rf.undisTrackedPoints[i]);
//            pt_idx.push_back(i);
//        }
//    }
//    cout << "Get points num: " << lp.size() << endl;
//
//    cout << "Set optimizer." << endl;
//    // Optimizer
//    g2o::SparseOptimizer optimizer;
//    
//    // linear solver
//    g2o::BlockSolver_6_3::LinearSolverType* linearSolver = new g2o::LinearSolverCSparse< g2o::BlockSolver_6_3::PoseMatrixType >();
//
//    // block solver
//    g2o::BlockSolver_6_3* block_solver = new g2o::BlockSolver_6_3( linearSolver );
//
//    // optimization algorithm
//    g2o::OptimizationAlgorithmLevenberg* algorithm = new g2o::OptimizationAlgorithmLevenberg( block_solver );
//
//    optimizer.setAlgorithm( algorithm );
//    optimizer.setVerbose( false );
//
//    cout << "add pose vertices." << endl;
//    // add pose vertices
//    g2o::VertexSE3Expmap* v1 = new g2o::VertexSE3Expmap();
//    v1->setId(0);
//    v1->setFixed(true);
//    v1->setEstimate(g2o::SE3Quat());
//    optimizer.addVertex(v1);
//
//    g2o::VertexSE3Expmap* v2 = new g2o::VertexSE3Expmap();
//    v2->setId(1);
//    v2->setEstimate(g2o::SE3Quat());
//    optimizer.addVertex(v2);
//
//    cout << "add 3d points vertices" << endl;
//    // add 3d point vertices
//    for (int i = 0, _end = (int)lp.size(); i < _end; i++) {
//        g2o::VertexSBAPointXYZ* v = new g2o::VertexSBAPointXYZ();
//        v->setId(2+i);
//        double z = 10;
//        double x = ( lp[i].x - K->cx ) / K->fx * z;
//        double y = ( lp[i].y - K->cy ) / K->fy * z;
//        v->setMarginalized(true);
//        v->setEstimate( Eigen::Vector3d(x, y, z) );
//        optimizer.addVertex( v );
//    }
//
//    cout << "add camera parameters" << endl;
//    // prepare camera parameters
//    g2o::CameraParameters* camera = new g2o::CameraParameters( (K->fx + K->fy)/2, Eigen::Vector2d(K->cx, K->cy), 0 );
//    camera->setId(0);
//    optimizer.addParameter(camera);
//
//    cout << "add edges" << endl;
//    // prepare edges
//    vector< g2o::EdgeProjectXYZ2UV* > edges;
//    for (int i = 0, _end = (int)lp.size(); i < _end; i++) {
//        g2o::EdgeProjectXYZ2UV* edge = new g2o::EdgeProjectXYZ2UV();
//        edge->setVertex( 0, dynamic_cast< g2o::VertexSBAPointXYZ* > (optimizer.vertex(i+2)) );
//        edge->setVertex( 1, dynamic_cast< g2o::VertexSE3Expmap* > (optimizer.vertex(0)) );
//
//        edge->setMeasurement( Eigen::Vector2d(lp[i].x, lp[i].y) );
//        edge->setInformation( Eigen::Matrix2d::Identity() );
//        edge->setParameterId(0, 0);
//
//        edge->setRobustKernel( new g2o::RobustKernelHuber() );
//        optimizer.addEdge( edge );
//        edges.push_back( edge );
//    }
//
//    for (int i = 0, _end = (int)rp.size(); i < _end; i++) {
//        g2o::EdgeProjectXYZ2UV* edge = new g2o::EdgeProjectXYZ2UV();
//        edge->setVertex( 0, dynamic_cast< g2o::VertexSBAPointXYZ* > (optimizer.vertex(i+2)) );
//        edge->setVertex( 1, dynamic_cast< g2o::VertexSE3Expmap* > (optimizer.vertex(1)) );
//
//        edge->setMeasurement( Eigen::Vector2d(rp[i].x, rp[i].y) );
//        edge->setInformation( Eigen::Matrix2d::Identity() );
//        edge->setParameterId(0, 0);
//
//        edge->setRobustKernel( new g2o::RobustKernelHuber() );
//        optimizer.addEdge( edge );
//        edges.push_back( edge );
//    }
//
//    cout << "optimization" << endl;
//    // optimization
//    optimizer.setVerbose(true);
//    optimizer.initializeOptimization();
//    optimizer.optimize(100);
//
//    // SE3 estimate
//    g2o::VertexSE3Expmap* v = 
//            dynamic_cast< g2o::VertexSE3Expmap* >( optimizer.vertex(1) );
//    Eigen::Isometry3d pose = v->estimate();
//    // set optimization pose result, do not forget this step
//    Eigen::Matrix4d T = pose.matrix();
//    cv::Mat res;
//    cv::eigen2cv(T, res);
//    cv::Mat R = res.rowRange(0,3).colRange(0,3);
//    cv::Mat t = res.rowRange(0,3).col(3);
//
//    cv::Mat Rt = R.t();
//    cv::Mat _t = -Rt*t;
//
//    mR = Rt.clone();
//    mt = _t.clone();
//
//    rf.R = mR.clone();
//    rf.t = mt.clone();
//
//    cout << "mR: " << endl;
//    cout << mR <<  endl;
//
//    cout << "mt: " << endl;
//    cout << mt << endl;
//
//    // points estimate
//    for (int i = 0, _end = (int)lp.size(); i < _end; i++ ) {
//        g2o::VertexSBAPointXYZ* v = 
//                dynamic_cast< g2o::VertexSBAPointXYZ* > 
//                ( optimizer.vertex(i+2) );
//        Eigen::Vector3d pos = v->estimate();
//        // set Mapping points and lf points
//        cv::Point3f* mpt = new cv::Point3f(pos[0], pos[1], pos[2]);
//        map->mapPoints.insert(mpt);
//        lf.map_2d_3d.insert(
//                Map_2d_3d_key_val(pt_idx[i], mpt));
//    }
//
//    // add keyFrame
//    map->keyFrames.push_back(&lf);
//
//    // num of inliers
//    int inliers = 0;
//    for (auto e:edges ) {
//        e->computeError();
//        if (e->chi2() > 1) {
//            cout << "error = " << e->chi2() << endl;
//        } else {
//            inliers++;
//        }
//    }
//    cout << "num inliers: " << inliers << endl;
//}

//void VisionTracker::TrackPose3D2D(const ImageFrame& lf, ImageFrame& rf)
//{
//    cv::Mat KMat = 
//            (cv::Mat_<double> (3,3) << K->fx, 0, K->cx, 0, K->fy, K->cy, 0, 0, 1);
//    vector< cv::Point2f > pts_2d;
//    vector< cv::Point3f > pts_3d;
//    pts_2d.reserve(lf.undisPoints.size());
//    pts_3d.reserve(lf.undisPoints.size());
//
//    const double *R_data = lf.R.ptr<double>(0);
//    const double *t_data = lf.t.ptr<double>(0);
//
//    {
//        std::lock_guard<std::mutex> lock(map->mMapMutex);
//        for (Map_2d_3d::const_iterator iter = lf.map_2d_3d.begin(), 
//                i_end = lf.map_2d_3d.end(); iter != i_end; iter++) {
//            if (rf.undisTrackedPoints[iter->left].x > 0) {
//                pts_2d.push_back(rf.undisTrackedPoints[iter->left]);    
//                cv::Point3f* pp = iter->right;
//                pts_3d.push_back(cv::Point3f(        // convert points from world to lf
//                            R_data[0]*pp->x + R_data[1]*pp->y + R_data[2]*pp->z + t_data[0],
//                            R_data[3]*pp->x + R_data[4]*pp->y + R_data[5]*pp->z + t_data[1],
//                            R_data[6]*pp->x + R_data[7]*pp->y + R_data[8]*pp->z + t_data[2]
//                            ));
//            }
//        }
//    }
//
//    if ( (int)pts_2d.size() < 10) {
//        printf("Tracked points less than 10, can not track pose 3d-2d!\n");
//        return;
//    }
//
//    //cv::Mat r, t, R, inliers;
//    //cv::solvePnPRansac (pts_3d, pts_2d, KMat, cv::Mat(), r, t, false, 100, 8.0, 0.99, inliers);   // opencv solvePnP result is bad, 
//    //cv::Rodrigues(r, R);                                                                          // do not use it
//
//    cv::Mat R,t;                            // set initial pose as the refImage
//    bundleAdjustment3D2D(pts_3d, pts_2d, KMat, R, t);    // optimize the pose by g2o
//
//    //cout << "BA:" << endl;
//    //cout << R << endl;
//    //cout << t << endl;
//    
//    // use Median filter to make the result stable
//    //TooN::SO3<> so3;
//    //Converter::Mat_TooNSO3(R, so3);
//    //TooN::Vector<3> w = so3.ln(); 
//    //w[0] = medianFilter[0].filterAdd(w[0]);
//    //w[1] = medianFilter[1].filterAdd(w[1]);
//    //w[2] = medianFilter[2].filterAdd(w[2]);
//    //t.at<double>(0) = medianFilter[3].filterAdd(t.at<double>(0));
//    //t.at<double>(1) = medianFilter[4].filterAdd(t.at<double>(1));
//    //t.at<double>(2) = medianFilter[5].filterAdd(t.at<double>(2));
//    //so3 = TooN::SO3<>::exp(w);
//    //Converter::TooNSO3_Mat(so3, R);
//
//    //cv::Mat Rt = R.t();
//    //cv::Mat _t = -Rt*t;
//
//    mR = R*lf.R;
//    mt = R*lf.t + t;
//
//    rf.R = mR.clone();
//    rf.t = mt.clone();
//
//}



//void VisionTracker::TriangulateNewPoints(ImageFrame& lf, ImageFrame& rf)
//{
//
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
//    bool checkPoints = initializer.CheckPoints(R, t, pts_4d);
//
//    if (!checkPoints) {
//        printf("Triangulate inliers num too small!\n");
//    }
//
//    for (int i = 0; i < pts_4d.cols; i++) {
//        float w = pts_4d.at<float>(3, i);
//        if (w!=0) {
//            cv::Point3f* mpt = new cv::Point3f(
//                    pts_4d.at<float>(0, i)/w,
//                    pts_4d.at<float>(1, i)/w,
//                    pts_4d.at<float>(2, i)/w);
//
//            cout << *mpt << endl;
//
//            map->mapPoints.insert(mpt);     // Insert map point pointer to std::set
//            lf.map_2d_3d.insert(  // Insert bimap key-val to boost::bimap in lf
//                    Map_2d_3d_key_val(pt_idx[i], mpt));
//        }
//    }
//
//}

//void VisionTracker::TrackPose3D2DHybrid(ImageFrame& lf, ImageFrame& rf)
//{
//    // prepare points, include 3d-2d pairs and 2d-2d pairs
//    vector< int > pt_idx;
//    vector< cv::Point3f* > pt_3d_ptr;
//    vector< cv::Point3f > pt_3d;
//    vector< cv::Point2f > pt_2d1, pt_2d2, pt_2d_a, pt_2d_b;
//
//    pt_3d_ptr.reserve(lf.map_2d_3d.size());
//    pt_3d.reserve(lf.map_2d_3d.size());
//    pt_2d1.reserve(lf.map_2d_3d.size());
//    pt_2d2.reserve(lf.map_2d_3d.size());
//
//    pt_idx.reserve(lf.points.size());
//    pt_2d_a.reserve(lf.points.size() - lf.map_2d_3d.size());
//    pt_2d_b.reserve(lf.points.size() - lf.map_2d_3d.size());
//
//    const double *R_data = lf.R.ptr<double>(0);
//    const double *t_data = lf.t.ptr<double>(0);
//
//    for (int i = 0, _end = (int)lf.points.size(); i < _end; ++i) {
//        if (rf.undisTrackedPoints[i].x > 0) {          // first should has crospondence
//            Map_2d_3d::left_const_iterator iter = 
//                    lf.map_2d_3d.left.find(i);
//            if (iter != lf.map_2d_3d.left.end()) {      // second should has 3d map point
//                cv::Point3f * pp = iter->second;
//                if (pp != NULL ) {
//                    pt_3d_ptr.push_back(pp);
//                    pt_3d.push_back(cv::Point3f(        // convert points from world to lf
//                                R_data[0]*pp->x + R_data[3]*pp->y + R_data[6]*pp->z - t_data[0],
//                                R_data[1]*pp->x + R_data[4]*pp->y + R_data[7]*pp->z - t_data[1],
//                                R_data[2]*pp->x + R_data[5]*pp->y + R_data[8]*pp->z - t_data[2]
//                                ));
//                    pt_2d1.push_back(lf.undisPoints[i]);
//                    pt_2d2.push_back(rf.undisTrackedPoints[i]);
//                }
//            } else {                                   // if no 3d point, add 2d-2d pair
//                pt_idx.push_back(i);
//                pt_2d_a.push_back(lf.undisPoints[i]);
//                pt_2d_b.push_back(rf.undisTrackedPoints[i]);
//            }
//        }
//    }
//
//    // setup g2o
//    // Optimizer
//    g2o::SparseOptimizer optimizer;
//    
//    // linear solver
//    g2o::BlockSolver_6_3::LinearSolverType* linearSolver = 
//            new g2o::LinearSolverCSparse< g2o::BlockSolver_6_3::PoseMatrixType >();
//
//    // block solver
//    g2o::BlockSolver_6_3* block_solver = new g2o::BlockSolver_6_3( linearSolver );
//
//    // optimization algorithm
//    g2o::OptimizationAlgorithmLevenberg* algorithm = 
//            new g2o::OptimizationAlgorithmLevenberg( block_solver );
//
//    optimizer.setAlgorithm( algorithm );
//    optimizer.setVerbose( false );
//
//
//    int vid = 0;
//    // add pose vertices
//    g2o::VertexSE3Expmap* v1 = new g2o::VertexSE3Expmap();
//    v1->setId(vid++);
//    v1->setFixed(true);
//    v1->setEstimate(g2o::SE3Quat());
//    optimizer.addVertex(v1);
//
//    g2o::VertexSE3Expmap* v2 = new g2o::VertexSE3Expmap();
//    v2->setId(vid++);
//    v2->setEstimate(g2o::SE3Quat());
//    optimizer.addVertex(v2);
//
//    // add 3d point vertices
//    // true 3d point
//    double ave_depth = 0;
//    for (int i = 0, _end = (int)pt_3d.size(); i < _end; i++) {
//        g2o::VertexSBAPointXYZ* v = new g2o::VertexSBAPointXYZ();
//        v->setId(vid++);
//        v->setMarginalized(true);
//        v1->setFixed(true);                            // existed points should be fixed
//        v->setEstimate( Eigen::Vector3d(pt_3d[i].x, pt_3d[i].y, pt_3d[i].z) );
//        optimizer.addVertex( v );
//
//        ave_depth += pt_3d[i].z;
//    }
//    ave_depth /= (double)pt_3d.size();
//
//    // fake 3d point
//    int fake3DIndex = vid;
//    for (int i = 0, _end = (int)pt_2d_a.size(); i < _end; i++) {
//        g2o::VertexSBAPointXYZ* v = new g2o::VertexSBAPointXYZ();
//        v->setId(vid++);
//        double z = ave_depth;
//        double x = ( pt_2d_a[i].x - K->cx ) / K->fx * z;
//        double y = ( pt_2d_a[i].y - K->cy ) / K->fy * z;
//        v->setMarginalized(true);
//        v->setEstimate( Eigen::Vector3d(x, y, z) );
//        optimizer.addVertex( v );
//    }
//
//    // prepare camera parameters
//    g2o::CameraParameters* camera = new g2o::CameraParameters( 
//            (K->fx + K->fy)/2, Eigen::Vector2d(K->cx, K->cy), 0 );
//    camera->setId(0);
//    optimizer.addParameter(camera);
//
//    // prepare edges
//    vector< g2o::EdgeProjectXYZ2UV* > edges;
//    // left
//    for (int i = 0, _end = (int)pt_3d.size(); i < _end; i++) {
//        g2o::EdgeProjectXYZ2UV* edge = new g2o::EdgeProjectXYZ2UV();
//        edge->setVertex( 0, 
//                dynamic_cast< g2o::VertexSBAPointXYZ* > (optimizer.vertex(i+2)) );
//        edge->setVertex( 1, dynamic_cast< g2o::VertexSE3Expmap* > (optimizer.vertex(0)) );
//
//        edge->setMeasurement( Eigen::Vector2d(pt_2d1[i].x, pt_2d1[i].y) );
//        edge->setInformation( Eigen::Matrix2d::Identity() );
//        edge->setParameterId(0, 0);
//
//        edge->setRobustKernel( new g2o::RobustKernelHuber() );
//        optimizer.addEdge( edge );
//        edges.push_back( edge );
//    }
//    for (int i = 0, _end = (int)pt_2d_a.size(); i < _end; i++) {
//        g2o::EdgeProjectXYZ2UV* edge = new g2o::EdgeProjectXYZ2UV();
//        edge->setVertex( 0, 
//                dynamic_cast< g2o::VertexSBAPointXYZ* >(optimizer.vertex(i+fake3DIndex)) );
//        edge->setVertex( 1, dynamic_cast< g2o::VertexSE3Expmap* > (optimizer.vertex(0)) );
//
//        edge->setMeasurement( Eigen::Vector2d(pt_2d_a[i].x, pt_2d_a[i].y) );
//        edge->setInformation( Eigen::Matrix2d::Identity() );
//        edge->setParameterId(0, 0);
//
//        edge->setRobustKernel( new g2o::RobustKernelHuber() );
//        optimizer.addEdge( edge );
//        edges.push_back( edge );
//    }
//
//    // right
//    for (int i = 0, _end = (int)pt_3d.size(); i < _end; i++) {
//        g2o::EdgeProjectXYZ2UV* edge = new g2o::EdgeProjectXYZ2UV();
//        edge->setVertex( 0, 
//                dynamic_cast< g2o::VertexSBAPointXYZ* > (optimizer.vertex(i+2)) );
//        edge->setVertex( 1, dynamic_cast< g2o::VertexSE3Expmap* > (optimizer.vertex(1)) );
//
//        edge->setMeasurement( Eigen::Vector2d(pt_2d2[i].x, pt_2d2[i].y) );
//        edge->setInformation( Eigen::Matrix2d::Identity() );
//        edge->setParameterId(0, 0);
//
//        edge->setRobustKernel( new g2o::RobustKernelHuber() );
//        optimizer.addEdge( edge );
//        edges.push_back( edge );
//    }
//    for (int i = 0, _end = (int)pt_2d_a.size(); i < _end; i++) {
//        g2o::EdgeProjectXYZ2UV* edge = new g2o::EdgeProjectXYZ2UV();
//        edge->setVertex( 0, 
//                dynamic_cast< g2o::VertexSBAPointXYZ* > (optimizer.vertex(i+fake3DIndex)) );
//        edge->setVertex( 1, dynamic_cast< g2o::VertexSE3Expmap* > (optimizer.vertex(1)) );
//
//        edge->setMeasurement( Eigen::Vector2d(pt_2d_b[i].x, pt_2d_b[i].y) );
//        edge->setInformation( Eigen::Matrix2d::Identity() );
//        edge->setParameterId(0, 0);
//
//        edge->setRobustKernel( new g2o::RobustKernelHuber() );
//        optimizer.addEdge( edge );
//        edges.push_back( edge );
//    }
//
//    // optimization
//    optimizer.setVerbose(true);
//    optimizer.initializeOptimization();
//    optimizer.optimize(50);   
//
//    // num of inliers
//    vector< int > inliers(edges.size()/2, 1);
//    int num_vert = (int)inliers.size();
//    for ( int i = 0, _end = (int)edges.size(); i < _end; i++) {
//        g2o::EdgeProjectXYZ2UV* e = edges[i];
//        e->computeError();
//        if (e->chi2() > 1) {
//            inliers[i%num_vert] = 0;
//            cout << "error = " << e->chi2() << endl;
//        } 
//    }
//    int num_inliers = 0;
//    for ( int i = 0, _end = (int)inliers.size(); i < _end; i++) {
//        num_inliers += inliers[i];
//    }
//    cout << "num inliers: " << num_inliers << endl;
//
////    // check inliers
////    double ratio_inlier = (double)num_inliers / (double)lp.size();
////    if (ratio_inlier < 0.8) {
////        printf("Inliers too small, less than 0.8 !\n");
////        return;
////    }
//
//    // SE3 estimate
//    g2o::VertexSE3Expmap* v = 
//            dynamic_cast< g2o::VertexSE3Expmap* >( optimizer.vertex(1) );
//    Eigen::Isometry3d pose = v->estimate();
//    // set optimization pose result, do not forget this step
//    Eigen::Matrix4d T = pose.matrix();
//    cv::Mat res;
//    cv::eigen2cv(T, res);
//    cv::Mat R = res.rowRange(0,3).colRange(0,3);
//    cv::Mat t = res.rowRange(0,3).col(3);
//    // pose composation
//    cv::Mat Rt = R.t();
//    cv::Mat _t = -Rt*t;
//    mR = lf.R * Rt;
//    mt = lf.R * _t + lf.t;
//    rf.R = mR.clone();
//    rf.t = mt.clone();
//
//    // points estimate
//    for (int i = 0, _end = (int)pt_3d.size() + (int)pt_2d_a.size(); i < _end; i++ ) {
//        if (inliers[i] == 0) {
//            if ( i < (int)pt_3d.size() )
//                printf("Warning: existed 3d points become outlier!\n");
//            continue;
//        }
//        g2o::VertexSBAPointXYZ* v = 
//                dynamic_cast< g2o::VertexSBAPointXYZ* > 
//                ( optimizer.vertex(i+2) );
//        Eigen::Vector3d pos = v->estimate();
//
//        // convert point from last keyframe to world
//        double nx = R_data[0]*pos[0] + R_data[1]*pos[1] + R_data[1]*pos[2] + t_data[0]; 
//        double ny = R_data[3]*pos[0] + R_data[4]*pos[1] + R_data[5]*pos[2] + t_data[1]; 
//        double nz = R_data[6]*pos[0] + R_data[7]*pos[1] + R_data[8]*pos[2] + t_data[2]; 
//
//        // set Mapping points and lf points
//        if (i < (int)pt_3d.size()) {
//            // update existed points
//            cv::Point3f* ep= pt_3d_ptr[i];
//            double distance = 
//                    (ep->x - nx)*(ep->x - nx) +
//                    (ep->y - ny)*(ep->y - ny) +
//                    (ep->z - nz)*(ep->z - nz);
//            distance = sqrt(distance);
//            printf("Update existed 3d points, distance of new and old: %f\n", distance);
//
//            // TODO::Update existed 3d points;
//            
//        } else {
//            // insert new points
//            cv::Point3f* mpt = new cv::Point3f(nx, ny, nz);
//            map->mapPoints.insert(mpt);
//            lf.map_2d_3d.insert(
//                    Map_2d_3d_key_val(pt_idx[i - (int)pt_3d.size()], mpt));
//        }
//    }
//
//}

//double VisionTracker::TrackFeatureOpticalFlow(ImageFrame& kf, ImageFrame& f)
//{
//    // optical flow fast feature by lastframe
//    f.opticalFlowTrackedFAST(lastFrame);
//    f.mRefFrame = &kf;
//
//    // check essentialmat with keyFrame
//    vector< int > pt_idx;
//    vector< cv::Point2f > pt_1, pt_2;
//    pt_idx.reserve(kf.points.size());
//    pt_1.reserve(kf.points.size());
//    pt_2.reserve(kf.points.size());
//
//    for (int i = 0, _end = (int)f.trackedPoints.size(); i < _end; i++) {
//        if (f.trackedPoints[i].x > 0) {
//            pt_1.push_back(kf.undisPoints[i]);
//            pt_2.push_back(f.undisTrackedPoints[i]);
//            pt_idx.push_back(i);
//        }
//    }
//
//    // essential matrix estimation validation
//    cv::Mat inlier;
//    TIME_BEGIN()
//    cv::findEssentialMat(pt_1, pt_2, 
//            (K->fx + K->fy)/2, cv::Point2d(K->cx, K->cy),
//            cv::RANSAC, 0.9999, 2, inlier);
//    TIME_END("essential matrix estimation")
//
//    int num_inliers = 0;
//    for (int i = 0, _end = (int)pt_1.size(); i < _end; i++) {
//        if (inlier.at< unsigned char >(i) == 0) {
//            int idx = pt_idx[i];
//            f.trackedPoints[idx].x = f.trackedPoints[idx].y = 0;
//            f.undisTrackedPoints[idx].x = f.undisTrackedPoints[idx].y = 0;
//        } else {
//            num_inliers++;
//        }
//    }
//
//    // upate last frame
//    lastFrame = f;
//
//    return (double)num_inliers / (double)kf.points.size();
//
//}

//void VisionTracker::updateRefFrame(ImageFrame* kf)
//{
//    std::lock_guard<std::mutex> lock(mRefFrameMutex);
//    refFrame = kf;
//    lastFrame = *kf;
//}

void VisionTracker::AddTrace(ImageFrame& f)
{
    RSequence.push_back(f.R.clone());
    tSequence.push_back(f.t.clone());
}

cv::Mat VisionTracker::GetTwcMatNow()
{

    if (mR.empty()||mt.empty())
        return cv::Mat();

    cv::Mat res = cv::Mat::eye(4, 4, CV_64FC1);
    mR.copyTo(res.rowRange(0,3).colRange(0,3));
    mt.copyTo(res.rowRange(0,3).col(3));
    return res;
}

cv::Mat VisionTracker::GetTcwMatNow()
{

    if (mR.empty()||mt.empty())
        return cv::Mat();

    cv::Mat res = cv::Mat::eye(4, 4, CV_64FC1);
    cv::Mat Rt = mR.t();
    cv::Mat _t = -Rt*mt;
    Rt.copyTo(res.rowRange(0,3).colRange(0,3));
    _t.copyTo(res.rowRange(0,3).col(3));
    return res;
}

vector< cv::Mat > VisionTracker::GetTcwMatSequence()
{
   if (RSequence.empty() || tSequence.empty()) 
       return vector< cv::Mat >();

   vector< cv::Mat > resv;
   for (int i = 0, _end = (int)RSequence.size(); i < _end; i++) {
       cv::Mat res = cv::Mat::eye(4, 4, CV_64FC1);
       cv::Mat Rt = RSequence[i].t();
       cv::Mat _t = -Rt*tSequence[i];
       Rt.copyTo(res.rowRange(0,3).colRange(0,3));
       _t.copyTo(res.rowRange(0,3).col(3));
       resv.push_back(res);
   }

   return resv;
}

double VisionTracker::ZMSSD(cv::Mat& img1, cv::Point2f& pt1, int level1, 
                     cv::Mat& img2, cv::Point2f& pt2, int level2, 
                     cv::Mat &wrapM) 
{
    double scale1 = (1 << level1);
    double scale2 = (1 << level2);

    double *W = wrapM.ptr<double>(0);

    // mean 1
    double mean1 = 0;
    int count = 0;
    for (int dx = -3; dx < 4; dx++) {
        for (int dy = -3; dy < 4; dy++) {
            mean1 += img1.at<unsigned char>(pt1.y + dy*scale1, pt1.x + dx*scale1);
            count++;
        }
    }
    mean1 /= count;

    // mean 2
    double mean2 = 0;
    count = 0;
    for (int dx = -3; dx < 4; dx++) {
        for (int dy = -3; dy < 4; dy++) {
            double wdx = W[0] * dx + W[1] * dy;
            double wdy = W[2] * dx + W[3] * dy;
            mean2 += img2.at<unsigned char>(pt2.y + wdy*scale2, pt2.x + wdx*scale2);
            count++;
        }
    }
    mean2 /= 49;

    // SSD 
    double SSD = 0;
    for (int dx = -3; dx < 4; dx++) {
        for (int dy = -3; dy < 4; dy++) {
            double wdx = W[0] * dx + W[1] * dy;
            double wdy = W[2] * dx + W[3] * dy;
            double pix1 = img1.at<unsigned char>(pt1.y + dy*scale1, pt1.x + dx*scale1) - mean1;
            double pix2 = img2.at<unsigned char>(pt2.y + wdy*scale2, pt2.x + wdx*scale2) - mean2;
            SSD = SSD + (pix2 - pix1)*(pix2 - pix1);
        }
    }

    return SSD;
}

double VisionTracker::SSD(cv::Mat& img1, cv::Point2f& pt1, int level1, 
                     cv::Mat& img2, cv::Point2f& pt2, int level2, 
                     double *M) 
{
    double scale1 = (1 << level1);
    double scale2 = (1 << level2);

    // SSD 
    double SSD = 0;
    for (int dx = -3; dx < 4; dx++) {
        for (int dy = -3; dy < 4; dy++) {
            double wdx = M[0] * dx + M[1] * dy;
            double wdy = M[2] * dx + M[3] * dy;
            double pix1 = img1.at<unsigned char>
                    (pt1.y + dy*scale1, pt1.x + dx*scale1);
            double pix2 = img2.at<unsigned char>
                    (pt2.y + wdy*scale2, pt2.x + wdx*scale2);
            SSD = SSD + (pix2 - pix1)*(pix2 - pix1);
        }
    }

    return SSD;
}
