#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>


using namespace cv;
using namespace std;
using namespace cv::xfeatures2d;


class methods
{
public:
	void sift(Mat image);
	void harris(Mat image);
	void hough(Mat image);
	void bow();

private:
	Mat superposed_image, descriptor;
	vector<KeyPoint> keypoints;

	Ptr<SIFT> sift_detector = SIFT::create();
	Ptr<HarrisLaplaceFeatureDetector> harris_detector = HarrisLaplaceFeatureDetector::create();
};

