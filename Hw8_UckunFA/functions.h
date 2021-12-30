#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <fstream>


using namespace cv;
using namespace std;
using namespace cv::xfeatures2d;
using namespace cv::ml;

void dictionary();
void svm(int process);