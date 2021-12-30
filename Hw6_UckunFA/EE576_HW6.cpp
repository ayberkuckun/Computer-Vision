// EE576_HW6.cpp : This file contains the 'main' function. Program execution begins and ends there.
// Fehmi Ayberk Uçkun
// 2015401009

#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>
#include "dummy.h"


using namespace cv;
using namespace std;
using namespace cv::xfeatures2d;

int main()
{
    int a;
    string b;
    dummy x;

    cout << "Enter 1 to create the vocabluary, enter 0 if you already created it." << endl;
    cin >> a;

    if (a == 1)
    {
        x.dictionary();
    }

    else if (a == 0)
    {
        cout << "Choose to train or test. (train / test)" << endl;
        cin >> b;

        x.bow(b);
    }

    return 0;
}