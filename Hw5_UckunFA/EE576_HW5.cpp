// Fehmi Ayberk Uçkun
// 2015401009
// EE576 HW5

// EE576_HW5.cpp : This file contains the 'main' function. Program execution begins and ends there.

#include <iostream>
#include <opencv2/opencv.hpp>
#include "shape.h"
#include <opencv2/imgproc.hpp>
#include "EFD.h"

using namespace cv;
using namespace std;

int main()
{
    Mat image, gray_img;
    string question;

    image = imread("images/scene.jpg");           // Read image

    if (!image.data)                              // Check for invalid input
    {
        cout << "Could not open or find the image" << std::endl;
        return -1;
    }

    //resize(image, image, Size(225, 225));       // Optinal Resize for speed

    imshow("Image", image);
    waitKey(0);

    cvtColor(image, gray_img, CV_BGR2GRAY);

    cout << "Pick a Question! (q1 or q2)" << endl;
    cin >> question;

    if (question == "q1")
    {
        q1(image, gray_img);
    }

    else if (question == "q2")
    {
        ellipticFourierDescriptors(gray_img, image, 350);
    }

    return 0;
}