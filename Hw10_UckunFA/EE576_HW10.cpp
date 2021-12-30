// EE576_HW10.cpp : This file contains the 'main' function. Program execution begins and ends there.
// Fehmi Ayberk Uçkun
// 2015401009

#include "opencv2/calib3d.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/core/utility.hpp"
#include "opencv2/ximgproc.hpp"
#include <iostream>
#include <string>
#include <opencv2/opencv.hpp>
#include "functions.h"


using namespace cv;
using namespace std;

int main()
{
    int question;

    cout << "Choose Question! (1 or 2)" << endl;
    cin >> question;

    if (question == 1)
    {
        dense_stereo();
    }

    else if (question == 2)
    {
        epipolar();
    }

    return 0;
}