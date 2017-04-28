#include "Converter.h"

namespace Converter
{

    void TooNSO3_Mat(const TooN::SO3<>& so3, cv::Mat& mat)
    {
        if (mat.empty() || mat.rows != 3 || mat.cols != 3)
            mat = cv::Mat(3, 3, CV_64FC1);
        const TooN::Matrix<3,3>& M = so3.get_matrix();
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                mat.at<double>(i, j) = M[i][j];
            }
        }

    }

    void Mat_TooNSO3(const cv::Mat& mat, TooN::SO3<> & so3)
    {
        stringstream ss;
        ss << mat.at<double>(0,0) << " "
            << mat.at<double>(0,1) << " " 
            << mat.at<double>(0,2) << " " 
            << mat.at<double>(1,0) << " " 
            << mat.at<double>(1,1) << " " 
            << mat.at<double>(1,2) << " " 
            << mat.at<double>(2,0) << " " 
            << mat.at<double>(2,1) << " " 
            << mat.at<double>(2,2) ;
        ss >> so3;
    }

    cv::Point3f TransfromPoint3D(
            const cv::Point3f& P, const cv::Mat& R, const cv::Mat& t)
    {
        const double *R_data = R.ptr<double>(0);
        const double *t_data = t.ptr<double>(0);
        return cv::Point3f(
                R_data[0]*P.x + R_data[1]*P.y + R_data[2]*P.z + t_data[0],
                R_data[3]*P.x + R_data[4]*P.y + R_data[5]*P.z + t_data[1],
                R_data[6]*P.x + R_data[7]*P.y + R_data[8]*P.z + t_data[2]
                );
    }

    std::string getImageType(int number) 
    {
        // find type
        int imgTypeInt = number%8;
        std::string imgTypeString;

        switch (imgTypeInt)
        {
            case 0:
                imgTypeString = "8U";
                break;
            case 1:
                imgTypeString = "8S";
                break;
            case 2:
                imgTypeString = "16U";
                break;
            case 3:
                imgTypeString = "16S";
                break;
            case 4:
                imgTypeString = "32S";
                break;
            case 5:
                imgTypeString = "32F";
                break;
            case 6:
                imgTypeString = "64F";
                break;
            default:
                break;
        }

        // find channel
        int channel = (number/8) + 1;

        std::stringstream type;
        type<<"CV_"<<imgTypeString<<"C"<<channel;

        return type.str();
    }

}
