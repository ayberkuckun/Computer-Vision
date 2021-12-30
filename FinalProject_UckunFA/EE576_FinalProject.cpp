// EE576_FinalProject.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>
#include <opencv2/opencv.hpp>
#include "functions.h"

using namespace cv;
using namespace std;


int main()
{
	string dataset;
	int N;

	cout << "Choose The Dataset! (car or human)" << endl;
	cin >> dataset;
	cout << "Choose visible segment number(N)!" << endl;
	cin >> N;

	track(dataset, N);
}