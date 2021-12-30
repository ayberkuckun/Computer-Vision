// EE576_HW9.cpp : This file contains the 'main' function. Program execution begins and ends there.
// Fehmi Ayberk Uçkun
// 2015401009

#include <iostream>
#include <opencv2/opencv.hpp>
#include "functions.h"


using namespace cv;
using namespace std;

int main()
{
    string dataset, method;

    cout << "Choose an Optical Flow Method! (dense or sparse) " << endl;
    cin >> method;

    cout << "Choose a Dataset! (human or car) " << endl;
    cin >> dataset;
    cout << endl;

    if (method == "dense")
    {
        denseOF(dataset);
    }

    else if (method == "sparse")
    {
        sparseOF(dataset);
    }

    return 0;
}