#include "EFD.h"
#include "shape.h"

// Elliptic Fourier Descriptor analysis is done in this function.

double pi = 3.14159265358979323846;

void ellipticFourierDescriptors(Mat src,Mat originalImage,int harmonics)
{
	//namedWindow("src", CV_WINDOW_AUTOSIZE);
	//imshow("src", src);

	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;

	//Mat drawing1 = Mat::zeros(src.size(), CV_8UC1);
	Mat drawing1 = originalImage.clone();
	Mat drawing2 = originalImage.clone();
	Mat label_img = src.clone();
	//Mat drawing2 = Mat::zeros(src.size(), CV_8UC1);

	// Edges are detected using threshold using
	threshold(src, label_img, 100, 255, THRESH_BINARY);
	//Mat label_img = color_label(originalImage, src);
	
	// Contours are found
	findContours(label_img, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));

	// Approximate contours to polygons 
	vector<vector<Point> > contours_poly1(contours.size());
	vector<vector<Point> > contours_poly2(contours.size());

	for (int i = 0; i < contours.size(); i++)
	{
		approxPolyDP(Mat(contours[i]), contours_poly1[i], 3, true);
	}
	for (int i = 1; i < contours.size(); i++)
	{
		drawContours(drawing1, contours_poly1, i, Scalar(0, 0, 255), 2, 8, vector<Vec4i>(), 0, Point());
	}

	// Original contours are shown for comparison purposes
	imshow("Original contours", drawing1);
	waitKey(0);
	imwrite("originalcontours.jpeg", drawing1);
	// Moments are found. It calculates the contours regions and finds how many regions then calculates for moments.
     
	vector<Moments> mu(contours.size());
	for (int i = 0; i < contours.size(); i++)
	{
		mu[i] = moments(contours[i], false);
	}

	//  The mass center of each contour is found
	vector<Point2f> a0(contours.size());
	for (size_t i = 0; i < contours.size(); i++)
	{
		a0[i] = Point2f(mu[i].m10 / mu[i].m00, mu[i].m01 / mu[i].m00);
	}

	// The period and the number of harmonics are specified
	int T, P = harmonics;
	double **a, **b, **c, **d;

	//2D arrays are allocated for a,b,c,d variables of the elliptic Fourier descriptors analysis
	a = allocate2D(contours.size(), P + 1);
	b = allocate2D(contours.size(), P + 1);
	c = allocate2D(contours.size(), P + 1);
	d = allocate2D(contours.size(), P + 1);

	double w, dxt, dyt;
	double threshold = 0.1;
	int counter = 0;
	for (int i = 0; i < contours.size(); i++)
	{
		// If the size of the contour is below the threshold, the elliptic Fourier descriptors are not found for this tiny contour
		if (mu[i].m00 > threshold * originalImage.rows* originalImage.cols) // contour area function??
		{
			T = contours[i].size();
			w = 2 * pi / T;
			for (int tp = 1; tp < contours[i].size(); tp++)
			{
				for (int k = 1; k <= P; k++)
				{
					dxt = contours[i][tp].x - contours[i][tp - 1].x;
					dyt = contours[i][tp].y - contours[i][tp - 1].y;
					a[counter][k] += (1 / (pi*w*k*k)) * dxt * (cos(k*w*tp) - cos(k*w*(tp - 1)));
					b[counter][k] += (1 / (pi*w*k*k)) * dxt * (sin(k*w*tp) - sin(k*w*(tp - 1)));
					c[counter][k] += (1 / (pi*w*k*k)) * dyt * (cos(k*w*tp) - cos(k*w*(tp - 1)));
					d[counter][k] += (1 / (pi*w*k*k)) * dyt * (sin(k*w*tp) - sin(k*w*(tp - 1)));
				}
			}
			counter++;
		}
	}
	String ss;
	String harStr= "descriptors" + to_string(harmonics) + ".txt";

	ofstream descriptorsToTxt;
	descriptorsToTxt.open(harStr);



	descriptorsToTxt << "For harmoics: " + to_string(harmonics) << endl;
	descriptorsToTxt << "a:" << endl;
	for (int i = 0; i < counter; ++i){
		for (int j = 0; j < P; ++j){
			descriptorsToTxt << a[i][j] << ' ';
		}
		descriptorsToTxt << endl;
	}

	descriptorsToTxt << "b:" << endl;
	for (int i = 0; i < counter; ++i){
		for (int j = 0; j < P; ++j){
			descriptorsToTxt << b[i][j] << ' ';
		}
		descriptorsToTxt << endl;
	}

	descriptorsToTxt << "c:" << endl;
	for (int i = 0; i < counter; ++i){
		for (int j = 0; j < P; ++j){
			descriptorsToTxt << c[i][j] << ' ';
		}
		descriptorsToTxt << endl;
	}

	descriptorsToTxt << "d:" << endl;
	for (int i = 0; i < counter; ++i){
		for (int j = 0; j < P; ++j){
			descriptorsToTxt << d[i][j] << ' ';
		}
		descriptorsToTxt << endl;
	}
	descriptorsToTxt << "------" << endl;


	descriptorsToTxt.close();


	// Once all the needed variables are found, reconstruction starts from here
	counter = 0;
	for (int i = 0; i < contours.size(); i++)
	{
		if (mu[i].m00 > threshold* originalImage.rows* originalImage.cols)
		{
			a[i][0] = (double)a0[i].x;
			b[i][0] = (double)a0[i].y;
			T = contours[i].size();
			w = 2 * pi / T;

			for (int tp = 1; tp < contours[i].size(); tp++)
			{
				contours[i][tp].x = a0[i].x;
				contours[i][tp].y = a0[i].y;
			}
			for (int tp = 1; tp < contours[i].size(); tp++)
			{
				for (int k = 1; k <= P; k++)
				{
					contours[i][tp].x += a[counter][k] * cos(k*w*tp) + b[counter][k] * sin(k*w*tp);
					contours[i][tp].y += c[counter][k] * cos(k*w*tp) + d[counter][k] * sin(k*w*tp);
				}
			}
			counter++;
		}
	}


	// Approximate contours to polygons 
	vector<vector<Point> > contours_poly(contours.size());

	for (int i = 0; i < contours.size(); i++)
	{
		approxPolyDP(Mat(contours[i]), contours_poly2[i], 3, true);
	}

	// Contours are drawn and shown in a seperate window
	for (int i = 1; i < contours.size(); i++)
	{
		drawContours(drawing2, contours_poly2, i, Scalar(0,0,255), 2, 8, vector<Vec4i>(), 0, Point());
	}
	
	namedWindow("Harmonic " + to_string(harmonics), CV_WINDOW_AUTOSIZE);
	imshow("Harmonic " + to_string(harmonics), drawing2);
	waitKey(0);
	imwrite("Harmonic " + to_string(harmonics) + ".jpeg", drawing2);
}


//2D array is allocated using this function
double** allocate2D(int n, int m)
{
	double **array2D;

	array2D = (double**)calloc(n, sizeof(double *));

	for (size_t i = 0; i < n; i++)
	{
		array2D[i] = (double*)calloc(m, sizeof(double));
	}
	return array2D;
}
