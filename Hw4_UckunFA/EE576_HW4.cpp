// Fehmi Ayberk Uçkun
// 2015401009
// EE576 HW4

// EE576_HW4.cpp : This file contains the 'main' function. Program execution begins and ends there.

#include <iostream>
#include <opencv2/opencv.hpp>
#include "segmentation.h"
#include <opencv2/imgproc.hpp>

using namespace cv;
using namespace std;

int main()
{
    segmentation x;
    Mat image;
    string program;

    image = imread("images/seg5.jpg");           // Read image

    if (!image.data)                              // Check for invalid input
    {
        cout << "Could not open or find the image" << std::endl;
        return -1;
    }

    resize(image, image, Size(225, 225));       // Optinal Resize for speed

    imshow("Image", image);
    waitKey(0);

    cout << "Pick your program! (q1, q2.1 or q2.2)" << endl;
    cin >> program;

    if (program == "q1")
    {
        x.cc(image);
    }

    else if (program == "q2.1")
    {
        x.hsv_cc1(image);
    }

    else if (program == "q2.2")
    {
        x.hsv_cc2(image);
    }

    return 0;
}