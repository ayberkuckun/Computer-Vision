#include "methods.h"
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>


using namespace cv;
using namespace std;
using namespace cv::xfeatures2d;

// Sift method.
void methods::sift(Mat image)
{
	// Detect key points.
	sift_detector->detect(image, keypoints);

	// Draw points to the image.
	drawKeypoints(image, keypoints, superposed_image);

	// Show superposed image.
	imshow("SIFT", superposed_image);
	waitKey(0);
	destroyAllWindows();

	// Compute descriptor vectors.
	sift_detector->compute(image, keypoints, descriptor);

	// Write to the txt file.
	FileStorage file("SIFT.txt", FileStorage::WRITE);
	write(file, "keypoints", keypoints);
	write(file, "vectors", descriptor);
	file.release();
}

void methods::harris(Mat image)
{ 
	char corner[100];
	int t = 140;
	int a = 0;
	Mat har_image = Mat::zeros(image.size(), CV_32FC1);

	// Apply Harris Corner Detection
	cornerHarris(image, har_image, 2, 3, 0.04);

	Mat har_norm;
	Mat har_norm_scaled;

	// Process it for drawing
	normalize(har_image, har_norm, 0, 255, NORM_MINMAX, CV_32FC1, Mat());
	convertScaleAbs(har_norm, har_norm_scaled);
	cvtColor(har_norm_scaled, har_norm_scaled, COLOR_GRAY2BGR);

	// Open txt file for output
	FileStorage file("Harris.txt", FileStorage::WRITE);

	// Drawing circles
	for (int i = 0; i < har_norm.rows; i++)
	{
		for (int j = 0; j < har_norm.cols; j++)
		{
			if ((int)har_norm.at<float>(i, j) > t)
			{
				a = a + 1;
				circle(har_norm_scaled, Point(j, i), 5, Scalar(0, 0, 255), 2);

				// Write to the txt file.
				sprintf_s(corner, 100, "corner_%d", a);
				write(file, corner, Point(j, i));
			}
		}
	}

	file.release();

	// Show superposed image.
	imshow("Harris", har_norm_scaled);
	waitKey(0);
}

void methods::hough(Mat image)
{
	Mat edge_image;

	// Detect edges.
	Canny(image, edge_image, 50, 200, 3);

	// Show edge detected image.
	imshow("Canny", edge_image);
	waitKey();

	// Convert image back to bgr for further processing.
	cvtColor(edge_image, superposed_image, COLOR_GRAY2BGR);

	// Standard Hough Line Transform
	vector<Vec2f> lines;
	HoughLines(edge_image, lines, 1, CV_PI / 180, 100, 0, 0);
	
	// Draw the lines
	for (int i = 0; i < lines.size(); i++)
	{
		Point pt1, pt2;
		float rho = lines[i][0], theta = lines[i][1];
		double a = cos(theta), b = sin(theta);
		double x0 = a * rho, y0 = b * rho;

		pt1.x = cvRound(x0 + 1000 * (-b));
		pt1.y = cvRound(y0 + 1000 * (a));
		pt2.x = cvRound(x0 - 1000 * (-b));
		pt2.y = cvRound(y0 - 1000 * (a));
		line(superposed_image, pt1, pt2, Scalar(0, 0, 255), 3, LINE_AA);
	}

	// Show superposed image.
	imshow("Hough", superposed_image);
	waitKey();
	destroyAllWindows();

	// Write to the txt file.
	FileStorage file("Hough.txt", FileStorage::WRITE);
	write(file, "lines", lines);
	file.release();
}

void methods::bow()
{
#define DICTIONARY_BUILD 0 // Set to 1 for building vocabulary and 0 for testing.
#if DICTIONARY_BUILD == 1

	char filename[100];
	Mat input;
	Mat featuresUnclustered;

	// Image reading loop.

	for (int i = 0; i < 3; i++) 
	{
		for (int j = 1; j < 18; j++)
		{
			if (i == 0)
			{
				sprintf_s(filename, 100, "dataset/train/aeroplane/%d.jpg", j);
				input = imread(filename, CV_LOAD_IMAGE_GRAYSCALE);

				// Detection keypoints.
				sift_detector->detect(input, keypoints);

				// Compute descriptors.
				sift_detector->compute(input, keypoints, descriptor);

				// Put the all feature descriptors in a single Mat object.
				featuresUnclustered.push_back(descriptor);
			}
			
			else if(i == 1)
			{
				sprintf_s(filename, 100, "dataset/train/bicycle/%d.jpg", j);
				input = imread(filename, CV_LOAD_IMAGE_GRAYSCALE); 
				
				// Detection keypoints.
				sift_detector->detect(input, keypoints);
				
				// Compute descriptors.
				sift_detector->compute(input, keypoints, descriptor);

				// Put the all feature descriptors in a single Mat object. 
				featuresUnclustered.push_back(descriptor);
			}

			else if(i == 2)
			{
				sprintf_s(filename, 100, "dataset/train/car/%d.jpg", j);
				input = imread(filename, CV_LOAD_IMAGE_GRAYSCALE);   

				// Detection keypoints.
				sift_detector->detect(input, keypoints);

				// Compute descriptors.
				sift_detector->compute(input, keypoints, descriptor);

				// Put the all feature descriptors in a single Mat object. 
				featuresUnclustered.push_back(descriptor);
			}
		}
	}

	// Create the BoW (or BoF) trainer
	BOWKMeansTrainer bowTrainer(3);

	// Cluster the feature vectors
	Mat dictionary = bowTrainer.cluster(featuresUnclustered);

	// Store the vocabulary
	FileStorage fs("vocabulary.yml", FileStorage::WRITE);
	fs << "vocabulary" << dictionary;
	fs.release();

#else

	// Read the vocabulary    
	Mat dictionary;
	FileStorage fs("vocabulary.yml", FileStorage::READ);
	fs["vocabulary"] >> dictionary;
	fs.release();

	// Create a FLANN based matcher
	Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create(DescriptorMatcher::FLANNBASED);
	
	// Create Sift descriptor extractor
	Ptr<DescriptorExtractor> extractor = SIFT::create();
	
	// Create BoF (or BoW) descriptor extractor
	BOWImgDescriptorExtractor bowDE(extractor, matcher);
	
	// Set the dictionary
	bowDE.setVocabulary(dictionary);

	char filename[100];

	// Read the test image.
	sprintf_s(filename, 100, "images/test1.jpg");
	Mat img = imread(filename, CV_LOAD_IMAGE_GRAYSCALE);
	resize(img, img, Size(300, 200));

	// Detect key points.
	sift_detector->detect(img, keypoints);

	// Compute descriptor vectors.
	bowDE.compute(img, keypoints, descriptor);

	// Write to file.
	FileStorage fs1("histogram.yml", FileStorage::WRITE);
	fs1 << "histogram" << descriptor;
	fs1.release();

#endif   
}