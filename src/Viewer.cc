#include "Viewer.h"

void Viewer::run()
{
    pangolin::CreateWindowAndBind("Viewer", 1024, 768);

    glEnable(GL_DEPTH_TEST);

    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    // TODO::Create menu
    s_cam = new pangolin::OpenGlRenderState(
            pangolin::ProjectionMatrix(1024, 768, 500, 500, 
            512, 389, 0.1, 1000),
            pangolin::ModelViewLookAt(0, -5, -10, 0, 0, 0, 0, -1, 0)
            );

    d_cam = & ( pangolin::CreateDisplay()
        .SetBounds(0, 1, pangolin::Attach::Pix(175), 1.f, -1024.f/768.f)
        .SetHandler(new pangolin::Handler3D(*s_cam)) );

    M.SetIdentity();

    while(1) {
        glClear (GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        d_cam->Activate(*s_cam);
        glClearColor(1.f, 1.f, 1.f, 1.f);

        drawCameraNow();

        drawKeyFrames();

        drawMapPoints();

        drawMapAxis();

        pangolin::FinishFrame();

        if (checkFinished())
            break;
    }

}

void Viewer::init()
{
    pangolin::CreateWindowAndBind("Viewer", 1024, 768);

    glEnable(GL_DEPTH_TEST);

    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    // TODO::Create menu
    s_cam = new pangolin::OpenGlRenderState(
            pangolin::ProjectionMatrix(1024, 768, 500, 500, 
            512, 389, 0.1, 1000),
            pangolin::ModelViewLookAt(0, -5, -10, 0, 0, 0, 0, -1, 0)
            );

    d_cam = & ( pangolin::CreateDisplay()
        .SetBounds(0, 1, pangolin::Attach::Pix(175), 1.f, -1024.f/768.f)
        .SetHandler(new pangolin::Handler3D(*s_cam)) );

    M.SetIdentity();
}

void Viewer::requestDraw()
{
    glClear (GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    d_cam->Activate(*s_cam);
    glClearColor(1.f, 1.f, 1.f, 1.f);

    drawCameraNow();

    drawKeyFrames();

    drawMapPoints();

    drawMapAxis();

    pangolin::FinishFrame();
}

void Viewer::drawCameraNow()
{
    const float w = 1;
    const float h = w*0.75;
    const float z = w*0.6;

    if (tracker == NULL)
        return;

    if (!tracker->mR.empty()) {
        cv::Mat Tcw = tracker->GetTcwMatNow().t();

        glPushMatrix();

        glMultMatrixd(Tcw.ptr<double>(0));

        glLineWidth(2);
        glColor3f(0.0f,1.0f,0.0f);
        glBegin(GL_LINES);
        glVertex3f(0,0,0);
        glVertex3f(w,h,z);
        glVertex3f(0,0,0);
        glVertex3f(w,-h,z);
        glVertex3f(0,0,0);
        glVertex3f(-w,-h,z);
        glVertex3f(0,0,0);
        glVertex3f(-w,h,z);

        glVertex3f(w,h,z);
        glVertex3f(w,-h,z);

        glVertex3f(-w,h,z);
        glVertex3f(-w,-h,z);

        glVertex3f(-w,h,z);
        glVertex3f(w,h,z);

        glVertex3f(-w,-h,z);
        glVertex3f(w,-h,z);
        glEnd();

        glPopMatrix();
    }

}

void Viewer::drawKeyFrames()
{
    const float w = 1;
    const float h = w*0.75;
    const float z = w*0.6;


    if (map==NULL)
        return;
    //static int count = 0;

    //count++;
    //if (count == 100) {
    //    printf("Viewer: Draw %d keyframes\n", (int)map->keyFrames.size());
    //    count=0;
    //}
    
    for (std::set< ImageFrame* >::iterator iter = map->keyFrames.begin(), 
            i_end = map->keyFrames.end(); iter != i_end; iter++) {

        ImageFrame* pKF = *iter;
        cv::Mat Tcw = pKF->GetTcwMat().t();

        //if (count == 0) {
        //    cout << "Viewer: Draw keyframe " << i << " :" << endl;
        //    cout << Tcw << endl;
        //}

        glPushMatrix();

        glMultMatrixd(Tcw.ptr<double>(0));

        glLineWidth(2);
        glColor3f(0.0f,0.0f,1.0f);
        glBegin(GL_LINES);
        glVertex3f(0,0,0);
        glVertex3f(w,h,z);
        glVertex3f(0,0,0);
        glVertex3f(w,-h,z);
        glVertex3f(0,0,0);
        glVertex3f(-w,-h,z);
        glVertex3f(0,0,0);
        glVertex3f(-w,h,z);

        glVertex3f(w,h,z);
        glVertex3f(w,-h,z);

        glVertex3f(-w,h,z);
        glVertex3f(-w,-h,z);

        glVertex3f(-w,h,z);
        glVertex3f(w,h,z);

        glVertex3f(-w,-h,z);
        glVertex3f(w,-h,z);
        glEnd();

        glPopMatrix();

    }

}

void Viewer::drawMapPoints()
{

    if (map==NULL)
        return;

    glPointSize(3);
    glBegin(GL_POINTS);
    glColor3f(1.0,0.0,0.0);

    for (std::set< Measure3D* >::iterator s_it = map->mapPoints.begin(), 
            s_end = map->mapPoints.end(); s_it != s_end; s_it++) {
        Measure3D* pt = (*s_it);
        glVertex3f(pt->pt.x, pt->pt.y, pt->pt.z);
    }

    glEnd();

}

void Viewer::drawMapAxis()
{   
    glLineWidth(2);
    
    // Z Axis blue
    glColor3f(0.0f,0.0f,1.0f);
    glBegin(GL_LINES);
    glVertex3f(0,0,0);
    glVertex3f(0,0,1);
    glEnd();

    // Y Axis red
    glColor3f(1.0f,0.0f,0.0f);
    glBegin(GL_LINES);
    glVertex3f(0,0,0);
    glVertex3f(0,1,0);
    glEnd();

    // X Axis green
    glColor3f(0.0f,1.0f,0.0f);
    glBegin(GL_LINES);
    glVertex3f(0,0,0);
    glVertex3f(1,0,0);
    glEnd();
}

void Viewer::requestFinish()
{
    unique_lock<mutex> lock(mMutexFinish);
    isFinished = true;
}

bool Viewer::checkFinished()
{
    unique_lock<mutex> lock(mMutexFinish);
    return isFinished;
}
