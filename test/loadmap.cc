#include <stdio.h>
#include <stdlib.h>

#include <iostream>
#include <thread>

#include <opencv2/opencv.hpp>

#include "CameraDevice.h"
#include "ImageFrame.h"
#include "Mapping.h"
#include "Initializer.h"
#include "Viewer.h"
#include "Timer.h"

using namespace std;

int main(int argc, char** argv)
{

    Mapping mapping(NULL);
    mapping.Load(argv[1]);

    Viewer viewer(&mapping, NULL);
    std::thread* ptViewer = new std::thread(&Viewer::run, &viewer);

    ptViewer->join();

    return 0;
}
