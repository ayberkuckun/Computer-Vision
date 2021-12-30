#include "functions.h"

void image_read(vector<Mat>& images_right, vector<Mat>& images_left)
{
    vector<String> fn;
    Mat input;
    string path;

    for (int i = 0; i < 2; i++)
    {
        if (i == 0)
        {
            path = "images/SuburbanFollow-right_jpg/*.jpg";

            glob(path, fn);

            // Image reading loop.

            for (auto f : fn)
            {
                input = imread(f, IMREAD_GRAYSCALE);

                //resize(input, input, Size(), 0.5, 0.5);

                images_right.push_back(input);
            }
        }

        if (i == 1)
        {
            path = "images/SuburbanFollow-left_jpg/*.jpg";

            glob(path, fn);

            // Image reading loop.

            for (auto f : fn)
            {
                input = imread(f, IMREAD_GRAYSCALE);

                //resize(input, input, Size(), 0.5, 0.5);

                images_left.push_back(input);
            }
        }

    }
}

void dense_stereo()
{
    vector<Mat> images_left, images_right;

    image_read(images_right, images_left);

    Mat left_for_matcher;
    Mat right_for_matcher;

    Ptr<DisparityWLSFilter> wls_filter;

    int wsize = 21;
    int max_disp = 144;
    double lambda = 8000.0;
    double sigma = 1.5;
    double vis_mult = 1.0;

    Mat left_disp, right_disp;
    Mat filtered_disp, solved_disp, solved_filtered_disp;

    Ptr<StereoBM> left_matcher = StereoBM::create(max_disp, wsize);

    wls_filter = createDisparityWLSFilter(left_matcher);

    Ptr<StereoMatcher> right_matcher = createRightMatcher(left_matcher);

    for (int i = 0; i < images_left.size(); i++)
    {
        left_for_matcher = images_left[i];
        right_for_matcher = images_right[i];

        left_matcher->compute(left_for_matcher, right_for_matcher, left_disp);
        right_matcher->compute(right_for_matcher, left_for_matcher, right_disp);

        wls_filter->setLambda(lambda);
        wls_filter->setSigmaColor(sigma);
        wls_filter->filter(left_disp, left_for_matcher, filtered_disp, right_disp);

        Mat filtered_disp_vis;

        getDisparityVis(filtered_disp, filtered_disp_vis, vis_mult);

        double min;
        double max;

        minMaxIdx(filtered_disp_vis, &min, &max);

        filtered_disp_vis.convertTo(filtered_disp_vis, CV_8UC1, 255 / (max - min), -min);

        applyColorMap(filtered_disp_vis, filtered_disp_vis, COLORMAP_JET);

        Mat raw_disp_vis;

        getDisparityVis(left_disp, raw_disp_vis, vis_mult);

        resize(left_for_matcher, left_for_matcher, Size(), 0.5, 0.5);
        resize(right_for_matcher, right_for_matcher, Size(), 0.5, 0.5);
        resize(filtered_disp_vis, filtered_disp_vis, Size(), 0.5, 0.5);
        resize(raw_disp_vis, raw_disp_vis, Size(), 0.5, 0.5);

        namedWindow("filtered disparity", WINDOW_AUTOSIZE);
        imshow("filtered disparity", filtered_disp_vis);
        imwrite("images/filtered.jpg", filtered_disp_vis);

        namedWindow("raw disparity", WINDOW_AUTOSIZE);
        imshow("raw disparity", raw_disp_vis);
        imwrite("images/raw.jpg", raw_disp_vis);

        imshow("left image", left_for_matcher);
        imshow("right_image", right_for_matcher);

        char key = (char)waitKey();
        if (key == 27 || key == 'q' || key == 'Q')
        {
            break;
        }
    }
}

void sparseOF(Mat prvs, Mat next, vector<Point2f>& p00, vector<Point2f>& p11, vector<Scalar>& colors)
{
    Mat OPvectors1, OPvectors2;
    vector<Point2f> p0, p1;

    cvtColor(prvs, OPvectors2, CV_GRAY2BGR);
    cvtColor(next, OPvectors1, CV_GRAY2BGR);

    goodFeaturesToTrack(prvs, p0, 100, 0.1, 7, Mat(), 7, true, 0.1);

    vector<uchar> status;
    vector<float> err;

    TermCriteria criteria = TermCriteria((TermCriteria::COUNT) + (TermCriteria::EPS), 10, 0.03);
    calcOpticalFlowPyrLK(prvs, next, p0, p1, status, err, Size(15, 15), 2, criteria);

    cout << "Tracked Harris Feature List of the Current Frame" << endl << endl;

    int count;
    count = 0;

    for (int j = 0; j < p0.size(); j++)
    {
        if (status[j] == 1)
        {
            colors.push_back(Scalar((rand() & 255), (rand() & 255), (rand() & 255)));

            p00.push_back(p0[j]);
            p11.push_back(p1[j]);

            cout << p0[j] << endl;
            circle(OPvectors1, p1[j], 4, colors[count], -1);
            arrowedLine(OPvectors1, p1[j], p0[j], colors[count]);
            circle(OPvectors2, p0[j], 4, colors[count], -1);
            arrowedLine(OPvectors2, p0[j], p1[j], colors[count]);

            count++;
        }
    }

    resize(OPvectors2, OPvectors2, Size(), 0.5, 0.5);
    imshow("Vector Map on Left Image", OPvectors2);

    resize(OPvectors1, OPvectors1, Size(), 0.5, 0.5);
    imshow("Vector Map on Right Image", OPvectors1);
    waitKey();

    system("CLS");
}

void epipolar()
{
    vector<Mat> images_left, images_right;

    image_read(images_right, images_left);

    for (int i = 0; i < images_left.size(); i++)
    {
        vector<Point2f> p0, p1;
        vector<Scalar> colors;

        sparseOF(images_left[i], images_right[i], p0, p1, colors);

        Point2d temp1, temp2;
        vector<Point2d> p0d, p1d;

        for (int j = 0; j < p0.size(); j++) 
        {
            temp1 = static_cast<Point2d>(p0[j]);
            p0d.push_back(temp1);

            temp2 = static_cast<Point2d>(p1[j]);
            p1d.push_back(temp2);
        }

        Mat fundamentalMatrix = findFundamentalMat(p0d, p1d, FM_8POINT);
        vector<Vec3d> leftLines, rightLines;

        computeCorrespondEpilines(p0d, 1, fundamentalMatrix, rightLines);
        computeCorrespondEpilines(p1d, 2, fundamentalMatrix, leftLines);

        Mat leftImageRGB(images_left[i].size(), CV_8UC3);
        cvtColor(images_left[i], leftImageRGB, CV_GRAY2RGB);

        Mat rightImageRGB(images_right[i].size(), CV_8UC3);
        cvtColor(images_right[i], rightImageRGB, CV_GRAY2RGB);

        for (int j = 0; j < rightLines.size(); j++)
        {
            for (int k = 0; k < 2; k++)
            {
                double x0, y0, x1, y1;

                if (k == 0)
                {
                    Vec3d l = rightLines.at(j);
                    double a = l.val[0];
                    double b = l.val[1];
                    double c = l.val[2];

                    x0 = 0;
                    y0 = (-c - a * x0) / b;
                    x1 = rightImageRGB.cols;
                    y1 = (-c - a * x1) / b;

                    cout << "R_Feature Error: " << a * p1.at(j).x + b * p1.at(j).y + c << endl;
                    circle(rightImageRGB, p1[j], 4, colors[j], -1);
                    line(rightImageRGB, cvPoint(x0, y0), cvPoint(x1, y1), colors[j], 1);
                }

                else if (k == 1)
                {
                    Vec3d l = leftLines.at(j);
                    double a = l.val[0];
                    double b = l.val[1];
                    double c = l.val[2];

                    x0 = 0;
                    y0 = (-c - a * x0) / b;
                    x1 = rightImageRGB.cols;
                    y1 = (-c - a * x1) / b;

                    cout << "L_Feature Error: " << a * p0.at(j).x + b * p0.at(j).y + c << endl;
                    circle(leftImageRGB, p0[j], 4, colors[j], -1);
                    line(leftImageRGB, cvPoint(x0, y0), cvPoint(x1, y1), colors[j], 1);
                }
            }
        }

        //resize(rightImageRGB, rightImageRGB, Size(), 0.5, 0.5);
        imshow("Epipolar lines on right image", rightImageRGB);

        //resize(leftImageRGB, leftImageRGB, Size(), 0.5, 0.5);
        imshow("Epipolar lines on left image", leftImageRGB);
        waitKey();

        system("CLS");
    }
}
