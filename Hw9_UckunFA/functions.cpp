#include "functions.h"

vector<Mat> image_read(string dataset)
{
    vector<Mat> images;
    vector<String> fn;
    Mat input;
    char name[100];

    if (dataset == "car")
    {
        sprintf_s(name, 100, "images/Car2/img/*.jpg");
    }

    else if (dataset == "human")
    {
        sprintf_s(name, 100, "images/Human9/img/*.jpg");
    }

    glob(name, fn);

    // Image reading loop.

    for (auto f : fn)
    {
        input = imread(f, IMREAD_GRAYSCALE);

        images.push_back(input);
    }
    return images;
}

void denseOF(string dataset)
{
    vector<Mat> images;
    Mat prvs, next, OPvectors1, OPvectors2;

    images = image_read(dataset);

    for (int i = 1; i < images.size(); i++)
    {
        prvs = images[i - 1];
        next = images[i];

        OPvectors1 = Mat(prvs.size(), CV_8UC3, Scalar(255, 255, 255));
        cvtColor(prvs, OPvectors2, CV_GRAY2BGR);

        Mat flow(prvs.size(), CV_32FC2);
        calcOpticalFlowFarneback(prvs, next, flow, 0.5, 3, 15, 3, 5, 1.2, 0);

        Point p1, p2;

        for (int j = 0; j < prvs.rows; j+=10)
        {
            for (int k = 0; k < prvs.cols; k+=10)
            {
                p1 = Point2f(k, j);
                p2 = Point2f(k, j) + flow.at<Point2f>(j, k);

                arrowedLine(OPvectors1, p1, p2, Scalar(0, 0, 0));
                arrowedLine(OPvectors2, p1, p2, Scalar(0, 255, 0));
            }
        }

        imshow("Optical Flow Vectors1", OPvectors1);
        imshow("Optical Flow Vectors2", OPvectors2);

        int keyboard = waitKey(30);
        if (keyboard == 'q' || keyboard == 27)
            break;
        else if (keyboard == 'p')
        {
            waitKey(0);
        }
    }
}

void sparseOF(string dataset)
{
    vector<Mat> images;
    vector<Point2f> p0, p1;

    Mat prvs, next, OPvectors1, OPvectors2;

    images = image_read(dataset);

    for (int i = 1; i < images.size(); i++)
    {
        prvs = images[i - 1];
        next = images[i];

        OPvectors1 = Mat(prvs.size(), CV_8UC3, Scalar(255, 255, 255));
        cvtColor(prvs, OPvectors2, CV_GRAY2BGR);

        goodFeaturesToTrack(prvs, p0, 100, 0.3, 7, Mat(), 7, true, 0.04);

        vector<uchar> status;
        vector<float> err;

        TermCriteria criteria = TermCriteria((TermCriteria::COUNT) + (TermCriteria::EPS), 10, 0.03);
        calcOpticalFlowPyrLK(prvs, next, p0, p1, status, err, Size(15, 15), 2, criteria);

        cout << "Tracked Harris Feature List of the Current Frame" << endl << endl;

        for (int i = 0; i < p0.size(); i++)
        {
            if (status[i] == 1) 
            {
                cout << p0[i] << endl;

                arrowedLine(OPvectors1, p0[i], p1[i], Scalar(0, 0, 0));
                arrowedLine(OPvectors2, p0[i], p1[i], Scalar(0, 255, 0));
            }
        }

        imshow("Optical Flow Vectors1", OPvectors1);
        imshow("Optical Flow Vectors2", OPvectors2);

        int keyboard = waitKey(30);
        if (keyboard == 'q' || keyboard == 27)
            break;
        else if (keyboard == 'p')
        {
            waitKey(0);
        }

        system("CLS");
    }
}