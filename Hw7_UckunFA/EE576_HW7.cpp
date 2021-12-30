// EE576_HW7.cpp : This file contains the 'main' function. Program execution begins and ends there.
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
    int question;

    cout << "Enter question number!" << endl; 
    cin >> question;

    svm(question);

    return 0;
}