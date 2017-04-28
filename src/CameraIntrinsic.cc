#include "CameraIntrinsic.h"

CameraIntrinsic::CameraIntrinsic(double _cx, double _cy, double _fx, double _fy, double _k1,
                double _k2, int _width, int _height): cx(_cx), cy(_cy), fx(_fx), fy(_fy), k1(_k1), k2(_k2), width(_width), height(_height)
{
    printf("Init Camera Intrinsic:\n"
            "cx: %f, cy: %f, fx: %f, fy: %f, k1: %f, k2: %f, width: %d, height: %d\n",
            cx, cy, fx, fy, k1, k2, width, height);
}

CameraIntrinsic::CameraIntrinsic(const string& filename)
{
    loadFromFile(filename);
}

void CameraIntrinsic::loadFromFile(const string& filename)
{
    ifstream file(filename);
    if (!file.is_open()) {
        printf("error: Open Camera Intrinsic file %s failed, exit!\n", filename.c_str());
    }

    string line;
    getline(file, line);
    stringstream ss(line);
    ss >> fx >> fy >> cx >> cy >> k1 >> k2 >> width >> height;

    printf("Load Camera Intrinsic:\n"
            "cx: %f, cy: %f, fx: %f, fy: %f, k1: %f, k2: %f, width: %d, height: %d\n",
            cx, cy, fx, fy, k1, k2, width, height);

}

cv::Point2f CameraIntrinsic::distort(int x, int y) // TODO:: How to distort, inverse of undistort ? 
{

    double u = (x - cx) / fx;
    double v = (y - cy) / fy;

    double av = -2*k1;
    double bv = 1-2*k1*(u-v);
    double cv = -v-k1*(u*u+v*v)+2*k1*u*v;

    double nv = (-bv+sqrt(bv*bv-4*av*cv))/(2*av);
    
    double au = -2*k1;
    double bu= 1-2*k1*(v-u);
    double cu = -u-k1*(u*u+v*v)+2*k1*u*v;

    double nu = (-bu+sqrt(bu*bu-4*au*cu))/(2*au);

    nu = nu * fx + cx;
    nv = nv * fy + cy;

    return cv::Point2f(nu, nv);

}

cv::Point2f CameraIntrinsic::undistort(int x, int y) 
{
   double u = (x - cx) / fx;
   double v = (y - cy) / fy;
   double r2 = u*u + v*v;
   double r4 = r2*r2;

   double nu = u * (1 - k1*r2 - k2*r4) * fx + cx;
   double nv = v * (1 - k1*r2 - k2*r4) * fy + cy;

   //int nu = u * (1 - k1*r2 ) * fx + cx;
   //int nv = v * (1 - k1*r2 ) * fy + cy;

   return cv::Point2f(nu, nv);

}

cv::Point2f CameraIntrinsic::pixel2device(float x, float y)
{
    return cv::Point2f( (x-cx)/fx , (y-cy)/fy );
}

cv::Point2f CameraIntrinsic::device2pixel(float u, float v)
{
   return cv::Point2f( u*fx + cx, v*fy + cy ); 
}

cv::Point3f CameraIntrinsic::Proj2Dto3D(float x, float y, float d)
{
    float xx = d * (x - cx) / fx;
    float yy = d * (y - cy) / fy;
    return cv::Point3f(xx, yy, d);
}

cv::Point2f CameraIntrinsic::Proj3Dto2D(float x, float y, float z)
{
    float u = fx*x/z + cx;
    float v = fy*y/z + cy;
    return cv::Point2f(u, v);
}
