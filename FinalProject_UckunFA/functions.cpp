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
        input = imread(f);
        // not grayscale.

        images.push_back(input);
    }
    return images;
}

void buble_sort(Mat& sorted_stats, Mat& sorted_centroids)
{
    Mat temp_s, temp_c;

    for (int label1 = 1; label1 < sorted_stats.rows; label1++)
    {
        for (int label2 = label1; label2 < sorted_stats.rows; label2++)
        {
            if (sorted_stats.at<int>(label1, CC_STAT_AREA) < sorted_stats.at<int>(label2, CC_STAT_AREA))
            {
                sorted_stats.row(label1).copyTo(temp_s);
                sorted_stats.row(label2).copyTo(sorted_stats.row(label1));
                temp_s.copyTo(sorted_stats.row(label2));

                sorted_centroids.row(label1).copyTo(temp_c);
                sorted_centroids.row(label2).copyTo(sorted_centroids.row(label1));
                temp_c.copyTo(sorted_centroids.row(label2));

            }
        }
    }
}

void classic_segmentation(Mat image, Mat& sorted_stats, Mat& sorted_centroids, int N, string dataset)
{
    Mat gray_image;

    cvtColor(image, gray_image, COLOR_BGR2GRAY);

    Mat hist;
    equalizeHist(gray_image, hist);

    //imshow("Equalzied", hist);
    //waitKey();

    medianBlur(hist, hist, 3);
    //imshow("Blurred", hist);
    //waitKey();

    Mat kernel = (Mat_<float>(3, 3) <<
        1, 1, 1,
        1, -8, 1,
        1, 1, 1);

    Mat imgLaplacian;
    filter2D(hist, imgLaplacian, CV_32F, kernel);

    Mat sharp;
    gray_image.convertTo(sharp, CV_32F);
    Mat imgResult = sharp - imgLaplacian;

    // convert back to 8bits gray scale
    imgResult.convertTo(imgResult, CV_8UC3);
    imgLaplacian.convertTo(imgLaplacian, CV_8UC3);

    //imshow("Sharpened", imgResult);
    //waitKey();

    Mat binary_image;

    if (dataset == "human")
    {
        adaptiveThreshold(imgResult, binary_image, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY_INV, 255, 2);
    }

    else if (dataset == "car")
    {
        adaptiveThreshold(imgResult, binary_image, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY, 255, 2);
    }

    //threshold(imgResult, binary_image, 42, 255, THRESH_BINARY_INV | THRESH_OTSU);

    //imshow("Binary Image", binary_image);
    //waitKey();

    Mat dilated, eroted;
    Mat element = getStructuringElement(MORPH_RECT, Size(2, 2));

    erode(binary_image, eroted, Mat());
    //imshow("eroted", eroted);
    //waitKey();

    dilate(eroted, dilated, element);
    //imshow("eroted+Dilated", dilated);
    //waitKey();

    Mat labelImage, stats, centroids;

    int nLabels = connectedComponentsWithStats(dilated, labelImage, stats, centroids, 8, CV_32S);

    vector<Vec3b> colors(nLabels);
    colors[0] = Vec3b(0, 0, 0);

    for (int label = 1; label < nLabels; ++label)
    {
        colors[label] = Vec3b((rand() & 255), (rand() & 255), (rand() & 255));
    }

    Mat dst(gray_image.size(), CV_8UC3);

    for (int r = 0; r < dst.rows; ++r) {
        for (int c = 0; c < dst.cols; ++c) {
            //int label = labels.at<int>(r, c);
            int label = labelImage.at<int>(r, c);
            Vec3b& pixel = dst.at<Vec3b>(r, c);
            pixel = colors[label];
        }
    }

    imshow("Connected Components", dst);
    int keyboard = waitKey(300);
    if (keyboard == 'p')
    {
        waitKey(0);
    }

    buble_sort(stats, centroids);

    Mat temp_s = Mat(N + 1, stats.cols, stats.type(), stats.data);
    Mat temp_c = Mat(N + 1, centroids.cols, centroids.type(), centroids.data);

    temp_s.copyTo(sorted_stats);
    temp_c.copyTo(sorted_centroids);
}

void kmeans_segmentation(Mat image, Mat& sorted_stats, Mat& sorted_centroids, int N, string dataset)
{
    Mat gray_image, src;

    cvtColor(image, gray_image, COLOR_BGR2GRAY);
    image.copyTo(src);

    Mat labelImage(gray_image.size(), CV_32S);
    Mat binary_image = Mat::zeros(gray_image.size(), CV_8UC1);
    Mat last_image = Mat::zeros(gray_image.size(), CV_8UC1);
    Mat data;

    src.convertTo(data, CV_32F);
    data = data.reshape(1, data.total());

    // do kmeans
    Mat labels, centers, last;

    int kn;
    // increase means person segmented 3 part. walkway tile segments emphesized
    if (dataset == "car")
    {
        kn = 5;
    }
    else if (dataset == "human")
    {
        kn = 4;
    }

    kmeans(data, kn, labels, TermCriteria(CV_TERMCRIT_ITER, 10, 1.0), 3,
        KMEANS_PP_CENTERS, centers);

    labels = labels.reshape(1, src.rows);
    labels.convertTo(labels, CV_8UC1);

    vector<Mat> stats1, centroids1;
    int total = 0;

    for (int k = 0; k < kn; k++)
    {
        for (int i = 0; i < labels.rows; i++)
        {
            for (int j = 0; j < labels.cols; j++)
            {
                if (labels.at<uchar>(i, j) == k)
                {
                    binary_image.at<uchar>(i, j) = 255;
                }

                else
                {
                    binary_image.at<uchar>(i, j) = 0;
                }
            }
        }

        // Cool stuff
        //imshow("binary images", binary_image);
        //waitKey();

        Mat stats_temp, centroids_temp;

        int nLabels = connectedComponentsWithStats(binary_image, labelImage, stats_temp, centroids_temp, 8, CV_32S);

        /*
        double min, max;
        minMaxLoc(labelImage, &min, &max);

        for (int i = max; i < max + nLabels; i++)
        {
            for (int j = 0; j < labelImage.rows; j++)
            {
                for (int m = 0; m < labelImage.cols; m++)
                {
                    if (last_image.at<uchar>(j, m) != 0)
                    {
                        last_image.at<uchar>(j, m) = last_image.at<uchar>(j, m) + max;
                    }

                    else
                    {
                        last_image.at<uchar>(j, m) = labelImage.at<int>(j, m);
                    }
                }
            }
        }
        */

        for (int i = 1; i < nLabels; i++)
        {
            stats1.push_back(stats_temp.row(i));
            centroids1.push_back(centroids_temp.row(i));
        }

    }

    int nLabels = stats1.size();

    Mat stats(stats1.size(), 5, CV_32S);
    Mat centroids(centroids1.size(), 2, CV_32S);

    for (int i = 0; i < stats.rows; i++)
    {
        stats1[i].copyTo(stats.row(i));
        centroids1[i].copyTo(centroids.row(i));
    }

    labels.convertTo(labels, CV_32S);
    vector<Vec3b> colors(nLabels);
    colors[0] = Vec3b(0, 0, 0);

    for (int label = 1; label < nLabels; ++label)
    {
        colors[label] = Vec3b((rand() & 255), (rand() & 255), (rand() & 255));
    }

    Mat dst(gray_image.size(), CV_8UC3);

    for (int r = 0; r < dst.rows; ++r) {
        for (int c = 0; c < dst.cols; ++c) {
            int label = labels.at<int>(r, c);
            //int label = last_image.at<uchar>(r, c);
            Vec3b& pixel = dst.at<Vec3b>(r, c);
            pixel = colors[label];
        }
    }

    imshow("K-Means Labeled Segments", dst);
    int keyboard = waitKey(300);
    if (keyboard == 'p')
    {
        waitKey(0);
    }

    buble_sort(stats, centroids);

    Mat temp_s = Mat(N + 1, stats.cols, stats.type(), stats.data);
    Mat temp_c = Mat(N + 1, centroids.cols, centroids.type(), centroids.data);

    temp_s.copyTo(sorted_stats);
    temp_c.copyTo(sorted_centroids);

}

void track(string dataset, int N)
{
    vector<Mat> images;

    if (dataset == "car")
    {
        images = image_read("car");
    }

    else if (dataset == "human")
    {
        images = image_read("human");
    }

    Mat prvs, next;
    Mat rect_image1, rect_image2;
    Mat stats_p, stats_n;
    Mat centroids_p, centroids_n;

    char name[100];

    prvs = images[0];
    prvs.copyTo(rect_image1);

    string method;

    cout << "Inspect the segmentations! (Press anykey)" << endl << endl;

    kmeans_segmentation(prvs, stats_p, centroids_p, N, dataset);
    classic_segmentation(prvs, stats_p, centroids_p, N, dataset);
    cv::waitKey(0);

    cout << "Choose a segmentation type! (classic or kmeans)" << endl;
    cin >> method;

    if (method == "kmeans")
    {
        kmeans_segmentation(prvs, stats_p, centroids_p, N, dataset);
    }

    else
    {
        classic_segmentation(prvs, stats_p, centroids_p, N, dataset);
    }

    destroyAllWindows();

    vector<Point2f> p0;

    for (int j = 1; j < centroids_p.rows; j++)
    {
        p0.push_back(static_cast<Point2f>(centroids_p.row(j)));
    }

    Mat segment(N, 7, CV_32F);
    int total_seg = N;

    int lost_count = 0;
    Mat lost_segment(100, 7, CV_32F);;
    vector<Point2f> lost_c;

    for (int i = 0; i < N; i++)
    {
        segment.at<float>(i, 0) = i;
        segment.at<float>(i, 1) = 1;
        segment.at<float>(i, 2) = 1;
        segment.at<float>(i, 3) = p0[i].x;
        segment.at<float>(i, 4) = p0[i].y;
        segment.at<float>(i, 5) = segment.at<float>(i, 1);
        segment.at<float>(i, 6) = segment.at<float>(i, 2);
    }


    for (int label = 0; label < N; ++label)
    {
        drawMarker(rect_image1, p0[label], Scalar(0, 255, 0), 0, 10, 1.5, 8);
    }

    for (int i = 1; i < images.size(); i++)
    {
        vector<Point2f> n0, n0_approx;

        next = images[i];
        next.copyTo(rect_image2);

        if (method == "kmeans")
        {
            kmeans_segmentation(prvs, stats_n, centroids_n, N, dataset);
        }

        else
        {
            classic_segmentation(prvs, stats_n, centroids_n, N, dataset);
        }

        for (int j = 1; j < centroids_n.rows; j++)
        {
            n0.push_back(static_cast<Point2f>(centroids_n.row(j)));
        }

        vector<uchar> status;
        vector<float> err;

        TermCriteria criteria = TermCriteria((TermCriteria::COUNT) + (TermCriteria::EPS), 10, 0.03);
        calcOpticalFlowPyrLK(prvs, next, p0, n0_approx, status, err, Size(15, 15), 2, criteria);

        int segNo = 0;
        int same;
        double min;

        for (int j = 0; j < N; j++)
        {
            min = 99999.0;
            if (status[j] == 1)
            {
                segment.at<float>(j, 1) = 1;
                segment.at<float>(j, 2)++;
                segment.at<float>(j, 5) = n0_approx[j].x;
                segment.at<float>(j, 6) = n0_approx[j].y;

                for (int m = 0; m < N; m++)
                {
                    if (n0[m] != Point2f(0, 0))
                    {
                        if (norm(Mat(n0[m]), Mat(n0_approx[j]), NORM_L2) < min)
                        {
                            min = norm(Mat(n0[m]), Mat(n0_approx[j]), NORM_L2);
                            segNo = m;
                        }
                    }
                }

                // Control if it is one of the largest.
                if (min < 24)
                {
                    n0[segNo] = Point2f(0, 0);
                    arrowedLine(rect_image2, p0[j], n0_approx[j], Scalar(0, 0, 255), 1, 8, 0, 0.1);
                    drawMarker(rect_image2, n0_approx[j], Scalar(0, 255, 0), 0, 10, 1.5, 8);
                }

                else
                {
                    same = 0;
                    for (int m = 0; m < lost_count; m++)
                    {
                        if (segment.at<float>(j, 0) == lost_segment.at<float>(m, 0))
                        {
                            lost_segment.at<float>(m, 2) = segment.at<float>(j, 2);
                            lost_segment.at<float>(m, 5) = segment.at<float>(j, 5);
                            lost_segment.at<float>(m, 6) = segment.at<float>(j, 6);
                            same = 1;
                        }
                    }

                    if (same != 1)
                    {
                        lost_segment.at<float>(lost_count, 0) = segment.at<float>(j, 0);
                        lost_segment.at<float>(lost_count, 1) = segment.at<float>(j, 1);
                        lost_segment.at<float>(lost_count, 2) = segment.at<float>(j, 2);
                        lost_segment.at<float>(lost_count, 3) = segment.at<float>(j, 3);
                        lost_segment.at<float>(lost_count, 4) = segment.at<float>(j, 4);
                        lost_segment.at<float>(lost_count, 5) = segment.at<float>(j, 5);
                        lost_segment.at<float>(lost_count, 6) = segment.at<float>(j, 6);

                        lost_c.push_back(Point2f(lost_segment.at<float>(lost_count, 5), lost_segment.at<float>(lost_count, 6)));

                        lost_count++;
                    }

                    segment.at<float>(j, 1) = 0;
                }
            }

            else 
            {
                same = 0;
                for (int m = 0; m < lost_count; m++)
                {
                    if (segment.at<float>(j, 0) == lost_segment.at<float>(m, 0))
                    {
                        lost_segment.at<float>(m, 2) = segment.at<float>(j, 2);
                        lost_segment.at<float>(m, 5) = segment.at<float>(j, 5);
                        lost_segment.at<float>(m, 6) = segment.at<float>(j, 6);
                        same = 1;
                    }
                }

                if (same != 1)
                {
                    lost_segment.at<float>(lost_count, 0) = segment.at<float>(j, 0);
                    lost_segment.at<float>(lost_count, 1) = segment.at<float>(j, 1);
                    lost_segment.at<float>(lost_count, 2) = segment.at<float>(j, 2);
                    lost_segment.at<float>(lost_count, 3) = segment.at<float>(j, 3);
                    lost_segment.at<float>(lost_count, 4) = segment.at<float>(j, 4);
                    lost_segment.at<float>(lost_count, 5) = segment.at<float>(j, 5);
                    lost_segment.at<float>(lost_count, 6) = segment.at<float>(j, 6);

                    lost_c.push_back(Point2f(lost_segment.at<float>(lost_count, 5), lost_segment.at<float>(lost_count, 6)));

                    lost_count++;
                }

                segment.at<float>(j, 1) = 0;
            }
        }

        for (int j = 0; j < N; j++)
        {
            int stop = 0;

            if (n0[j] != Point2f(0,0))
            {
                for (int k = 0; k < lost_count; k++)
                {
                    if (norm(Mat(n0[j]), Mat(lost_c[k]), NORM_L2) < 24)
                    {
                        for (int m = 0; m < N; m++)
                        {
                            if (segment.at<float>(m, 1) == 0)
                            {
                                segment.at<float>(m, 0) = lost_segment.at<float>(k, 0);
                                segment.at<float>(m, 1) = lost_segment.at<float>(k, 1);
                                segment.at<float>(m, 2) = lost_segment.at<float>(k, 2) + 1;
                                segment.at<float>(m, 3) = lost_segment.at<float>(k, 3);
                                segment.at<float>(m, 4) = lost_segment.at<float>(k, 4);
                                segment.at<float>(m, 5) = n0[j].x;
                                segment.at<float>(m, 6) = n0[j].y;
                                stop = 1;

                                /*
                                if (dataset == "car")
                                {
                                    arrowedLine(rect_image2, Point2f(lost_segment.at<float>(k, 5), lost_segment.at<float>(k, 6)), n0[j], Scalar(0, 0, 255), 1, 8, 0, 0.1);
                                }
                                */

                                drawMarker(rect_image2, n0[j], Scalar(0, 255, 0), 0, 10, 1.5, 8);
                                break;
                            }
                        }

                        if (stop == 1)
                        {
                            break;
                        }
                    }
                }

                if (stop != 1)
                {
                    for (int m = 0; m < N; m++)
                    {
                        if (segment.at<float>(m, 1) == 0)
                        {
                            segment.at<float>(m, 0) = total_seg;
                            segment.at<float>(m, 1) = 1;
                            segment.at<float>(m, 2) = 1;
                            segment.at<float>(m, 3) = n0[j].x;
                            segment.at<float>(m, 4) = n0[j].y;
                            segment.at<float>(m, 5) = segment.at<float>(m, 3);
                            segment.at<float>(m, 6) = segment.at<float>(m, 4);
                            total_seg++;

                            drawMarker(rect_image2, Point2f(segment.at<float>(m, 5), segment.at<float>(m, 6)), Scalar(0, 255, 0), 0, 10, 1.5, 8);
                            break;
                        }
                    }
                }
            }
        }

        cout << i + 1 << ". Frame, " << N << " Largest Segments:" << endl << endl;

        for (int label = 0; label < N; ++label)
        {

            sprintf_s(name, 100, "Segment_No_%d", static_cast<int>(segment.at<float>(label, 0)));
            cout << name << endl;;

            if (static_cast<int>(segment.at<float>(label, 1)) == 1)
            {
                cout << "   " << "Current Visibility: " << "Yes" << endl;
            }

            else
            {
                cout << "   " << "Current Visibility: " << "No" << endl;
            }

            cout << "   " << "Number of Occurrences: " << static_cast<int>(segment.at<float>(label, 2)) << endl;
            cout << "   " << "Initial Coordinates: [" << static_cast<int>(segment.at<float>(label, 3)) << ", " << static_cast<int>(segment.at<float>(label, 4)) << "]" << endl;
            cout << "   " << "Current Coordinates: [" << static_cast<int>(segment.at<float>(label, 5)) << ", " << static_cast<int>(segment.at<float>(label, 6)) << "]" << endl;
            cout << endl;
        }
        
        resize(rect_image1, rect_image1, Size(640, 480));
        imshow("prvs", rect_image1);

        resize(rect_image2, rect_image2, Size(640, 480));
        imshow("next", rect_image2);

        if (i == 1)
        {
            waitKey();
        }
        
        int keyboard = waitKey(1000);
        if (keyboard == 'p')
        {
            waitKey(0);
        }

        next.copyTo(prvs);
        rect_image2.copyTo(rect_image1);

        for (int f = 0; f < N; f++)
        {
            p0[f] = Point2f(segment.at<float>(f, 5), segment.at<float>(f, 6));
        }

        //system("CLS");
    }
}