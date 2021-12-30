// EE576_HW3.cpp : This file contains the 'main' function. Program execution begins and ends there.
// Fehmi Ayberk Uçkun
// 2015401009

#include "methods.h"
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>


using namespace cv;
using namespace std;
using namespace cv::xfeatures2d;

int main()
{
    int question;
    string method;
    Mat image;
    methods x;

    // Read the image.
    image = imread("images/2.png", CV_LOAD_IMAGE_GRAYSCALE);

    // Resize it.
    resize(image, image, Size(720, 720));

    imshow("Original", image);
    waitKey(0);

    // Pick a question.
    cout << "Choose a question. (1, 2 or 3)" << endl;
    cin >> question;

    if (question == 1)
    {
        cout << "Harris or SIFT?" << endl;
        cin >> method;

        if (method == "SIFT")
        {
            x.sift(image);
        }

        else if(method == "Harris")
        {
            x.harris(image);
        }
    }

    else if (question == 2)
    {
        x.hough(image);
    }

    else if (question == 3)
    {
        x.bow();
    }

    return 0;
}