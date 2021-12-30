// EE576_HW8.cpp : This file contains the 'main' function. Program execution begins and ends there.
// Fehmi Ayberk Uçkun
// 2015401009

#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>
#include "functions.h"


using namespace cv;
using namespace std;
using namespace cv::xfeatures2d;

int main()
{
    int a;
    int process;

    cout << "Enter Process number!" << endl;
    cin >> process;

    if (process == 1)
    {
        dictionary();
    }

    else
    {
        svm(process);
    }

    return 0;
}