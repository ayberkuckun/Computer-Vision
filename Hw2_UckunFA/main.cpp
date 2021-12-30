// Fehmi Ayberk Uçkun
// 2015401009
// EE576 HW2

// EE576_HW2.cpp : This file contains the 'main' function. Program execution begins and ends there.

#include <iostream>
#include <opencv2/opencv.hpp>
#include "image_proc.h"

using namespace cv;
using namespace std;


int main()
{
    image_proc exp;
    string space;
    Mat image;

	image = imread("images/object.jpg");           // Read image

    if (!image.data)                              // Check for invalid input
    {
        cout << "Could not open or find the image" << std::endl;
        return -1;
    }

    while (1)
    {
        cout << "Specify the operation! (rgb, hsv or gaussian)" << endl;
        cin >> space;

        if (space == "rgb")
        {
            exp.proc_rgb(image);            // RGB colorspace operations 
            return 0;
        }
        else if (space == "hsv")
        {
            exp.proc_hsi(image);            // HSV colorspace operations
            return 0;
        }
        else if (space == "gaussian")
        {
            exp.intensity_op(image);            // Gaussian operation
            return 0;
        }
        else
        {
            cout << "Try again" << endl;
        }
    }
    return 0;
}
