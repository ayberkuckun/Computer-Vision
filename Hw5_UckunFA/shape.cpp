#include <iostream>
#include <opencv2/opencv.hpp>
#include "shape.h"
#include <opencv2/imgproc.hpp>

using namespace cv;
using namespace std;

int colors[10][6] = {
    0, 0, 210, 180, 70, 255,        // White
    0, 0, 0, 180, 255, 25,          // Black
    0, 0, 50, 180, 30, 210,         // Grey
    15, 80, 80, 44, 255, 255,       // Yellow
    45, 80, 80, 74, 255, 255,       // Green
    75, 80, 80, 104, 255, 255,      // Cyan
    105, 80, 80, 134, 255, 255,     // Blue
    135, 80, 80, 164, 255, 255,     // Magenta
    165, 80, 80, 180, 255, 255,     // Red1
    0, 80, 80, 14, 255, 255,        // Red2
};

double thr = 0.01;

void q1(Mat image, Mat gray_img)
{
    Mat hsv, binary_img, binary_img2;
    Mat drawing1 = image.clone();
    Mat rect_img;

    cvtColor(gray_img, rect_img, COLOR_GRAY2BGR);

    vector<vector<Point> > contours;
    vector<Vec4i> hierarchy;

    cvtColor(image, hsv, COLOR_BGR2HSV);

    FileStorage fs("Segments.txt", FileStorage::WRITE);

    int no = 0;

    for (int i = 0; i < 9; i++)
    {
        inRange(hsv, Scalar(colors[i][0], colors[i][1], colors[i][2]), Scalar(colors[i][3], colors[i][4], colors[i][5]), binary_img);

        if (i == 8)
        {
            inRange(hsv, Scalar(colors[i+1][0], colors[i+1][1], colors[i+1][2]), Scalar(colors[i+1][3], colors[i+1][4], colors[i+1][5]), binary_img2);
            binary_img = binary_img | binary_img2;
        }

        findContours(binary_img, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));

        /// Get the moments
        vector<Moments> mu(contours.size());
        for (int j = 0; j < contours.size(); j++)
        {
            mu[j] = moments(contours[j], false);
        }

        char name[100];

        ///  Get the mass centers:
        vector<Point2f> mc(contours.size());
        for (int k = 0; k < contours.size(); k++)
        {
            if (mu[k].m00 > thr * image.rows * image.cols)
            {
                mc[k] = Point2f(mu[k].m10 / mu[k].m00, mu[k].m01 / mu[k].m00);

                sprintf_s(name, 100, "Segment No_%d", no);

                no++;

                fs << name;
                fs << "{" << "Area" << mu[k].m00;
                fs << "Center" << mc[k];
                fs << "Second Order Moments";
                fs << "{" << "mu11" << mu[k].mu11;
                fs << "mu12" << mu[k].mu12;
                fs << "mu21" << mu[k].mu21;
                fs << "mu20" << mu[k].mu20;
                fs << "mu02" << mu[k].mu02;
                fs << "mu30" << mu[k].mu30;
                fs << "mu03" << mu[k].mu03 << "}" << "}";

                Rect rect = boundingRect(contours[k]);
                Scalar a = Scalar((rand() & 255), (rand() & 255), (rand() & 255));
                rectangle(rect_img, rect, a, 2);
                drawMarker(rect_img, mc[k], a, 0, 7, 2, 8);
            }
        }
    }

    imshow("rect", rect_img);
    waitKey(0);

    fs.release();
}