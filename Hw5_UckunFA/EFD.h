#pragma once
#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <fstream>

using namespace std;
using namespace cv;

void ellipticFourierDescriptors(Mat src, Mat originalImage, int harmonics);
double** allocate2D(int n, int m);
