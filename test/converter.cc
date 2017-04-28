#include "Converter.h"

int main()
{
    TooN::SO3<> so3 = TooN::SO3<>::exp(TooN::makeVector(1,2,3)); 
    cv::Mat mat;
    Converter::TooNSO3_Mat(so3, mat);
    cout << "so3: " <<  endl;
    cout << so3 << endl;
    cout << "to mat: " << endl;
    cout << mat << endl;

    Converter::Mat_TooNSO3(mat, so3);
    cout << "mat: " << endl;
    cout << mat << endl;
    cout << "to so3: " << endl;
    cout << so3 << endl;

    return 0;
}

