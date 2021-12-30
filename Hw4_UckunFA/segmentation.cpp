#include <iostream>
#include <opencv2/opencv.hpp>
#include "segmentation.h"
#include <opencv2/imgproc.hpp>

using namespace cv;
using namespace std;


void segmentation::cc(Mat image)
{
    Mat gray_image;
    Mat binary_image;
    Mat rect_image;

    char name[100];

    cvtColor(image, gray_image, COLOR_BGR2GRAY);

    binary_image = binary_thresh < 128 ? (gray_image < binary_thresh) : (gray_image > binary_thresh);

	Mat stats, centroids;
    Mat labelImage(gray_image.size(), CV_32S);
    int bigger_segN = 0;

    cvtColor(gray_image, rect_image, COLOR_GRAY2BGR);

    int nLabels = connectedComponentsWithStats(binary_image, labelImage, stats, centroids, 8, CV_32S);

    std::vector<Vec3b> colors(nLabels);
    colors[0] = Vec3b(0, 0, 0);

    for (int label = 1; label < nLabels; ++label)
    {
            colors[label] = Vec3b((rand() & 255), (rand() & 255), (rand() & 255));
    }

    Mat dst(gray_image.size(), CV_8UC3);

    for (int r = 0; r < dst.rows; ++r) {
        for (int c = 0; c < dst.cols; ++c) {
            int label = labelImage.at<int>(r, c);
            Vec3b& pixel = dst.at<Vec3b>(r, c);
            pixel = colors[label];
        }
    }

    imshow("Connected Components", dst);
    cv::waitKey(0);

    FileStorage file("segments.txt", FileStorage::WRITE);
    write(file, "Number of Generated Segments", nLabels);

for (int label = 1; label < nLabels; ++label)
{
    if (stats.at<int>(label, CC_STAT_AREA) > (gray_image.rows * gray_image.cols * size_thresh))
    {
        Vec2i center = centroids.row(label);
        bigger_segN++;
        sprintf_s(name, 100, "Segment_No_%d", label);
        file << name;
        file << "{" << "Center Point" << center;
        file << "Size" << stats.at<int>(label, CC_STAT_AREA) << "}";

        Rect rect(stats.at<int>(label, CC_STAT_LEFT), stats.at<int>(label, CC_STAT_TOP),
            stats.at<int>(label, CC_STAT_WIDTH), stats.at<int>(label, CC_STAT_HEIGHT));

        rectangle(rect_image, rect, Scalar(0, 0, 255));
        drawMarker(rect_image, center, Scalar(0, 255, 0), 0, 10, 1, 8);
    }

    else
    {
        colors[label] = Vec3b(0, 0, 0);
    }
}

write(file, "Total Number of Tresholded Segments", bigger_segN);

file.release();

Mat dst2(gray_image.size(), CV_8UC3);
for (int r = 0; r < dst2.rows; ++r) {
    for (int c = 0; c < dst2.cols; ++c) {
        int label = labelImage.at<int>(r, c);
        Vec3b& pixel = dst2.at<Vec3b>(r, c);
        pixel = colors[label];
    }
}
imshow("Connected Components Tresholded", dst2);
cv::waitKey(0);

imshow("rect", rect_image);
cv::waitKey(0);

}

void segmentation::hsv_cc1(Mat image)
{
    Mat stats, centroids;
    Mat rect_img;
    Mat hsv, gray_image;

    int i, j, k;
    int bigger_segN = 0;

    char name[1000];
    
    int nLabels;

    cvtColor(image, hsv, COLOR_BGR2HSV);
    cvtColor(image, gray_image, COLOR_BGR2GRAY);
    cvtColor(gray_image, rect_img, COLOR_GRAY2BGR);

    Mat labelImage(gray_image.size(), CV_32S);
    Mat binary_image = Mat::zeros(gray_image.size(), CV_8UC1);

    FileStorage file("segments2_1.txt", FileStorage::WRITE);

    for (i = 0; i < binary_image.rows; i++)
    {
        for (j = 0; j < binary_image.cols; j++)
        {
            if (hsv.at<Vec3b>(i, j)[2] > 128 && hsv.at<Vec3b>(i, j)[1] < 80)
            {
                binary_image.at<uchar>(i, j) = 1; // White
            }

            else
            {
                binary_image.at<uchar>(i, j) = 0;
            }
        }
    }

    nLabels = connectedComponentsWithStats(binary_image, labelImage, stats, centroids, 8, CV_32S);
 
    write(file, "Number of Generated Segments", nLabels);

    for (int label = 1; label < nLabels; ++label)
    {
        if (stats.at<int>(label, CC_STAT_AREA) > (gray_image.rows * gray_image.cols * size_thresh))
        {
            Vec2i center = centroids.row(label);
            bigger_segN++;

            sprintf_s(name, 1000, "Segment_No_%d", label);

            file << name;
            file << "{" << "Center Point" << center;
            file << "Size" << stats.at<int>(label, CC_STAT_AREA) << "}";

            Rect rect(stats.at<int>(label, CC_STAT_LEFT), stats.at<int>(label, CC_STAT_TOP),
                stats.at<int>(label, CC_STAT_WIDTH), stats.at<int>(label, CC_STAT_HEIGHT));

            rectangle(rect_img, rect, Scalar(0, 0, 255));
            drawMarker(rect_img, center, Scalar(0, 255, 0), 0, 10, 1, 8);
        }
    }

    write(file, "Total Number of Tresholded Segments", bigger_segN);

    ////////////////////////////////////////////////////////////////////

    bigger_segN = 0;

    for (i = 0; i < binary_image.rows; i++)
    {
        for (j = 0; j < binary_image.cols; j++)
        {
            if (hsv.at<Vec3b>(i, j)[2] < 30)
            {
                binary_image.at<uchar>(i, j) = 1; // Black
            }

            else
            {
                binary_image.at<uchar>(i, j) = 0;
            }
        }
    }

    nLabels = connectedComponentsWithStats(binary_image, labelImage, stats, centroids, 8, CV_32S);

    write(file, "Number of Generated Segments", nLabels);

    for (int label = 1; label < nLabels; ++label)
    {
        if (stats.at<int>(label, CC_STAT_AREA) > (gray_image.rows * gray_image.cols * size_thresh))
        {
            Vec2i center = centroids.row(label);
            bigger_segN++;

            sprintf_s(name, 1000, "Segment_No_%d", label);

            file << name;
            file << "{" << "Center Point" << center;
            file << "Size" << stats.at<int>(label, CC_STAT_AREA) << "}";

            Rect rect(stats.at<int>(label, CC_STAT_LEFT), stats.at<int>(label, CC_STAT_TOP),
                stats.at<int>(label, CC_STAT_WIDTH), stats.at<int>(label, CC_STAT_HEIGHT));

            rectangle(rect_img, rect, Scalar(0, 0, 255));
            drawMarker(rect_img, center, Scalar(0, 255, 0), 0, 10, 1, 8);
        }
    }

    write(file, "Total Number of Tresholded Segments", bigger_segN);

    ////////////////////////////////////////////////////////////////////

    bigger_segN = 0;

    for (i = 0; i < binary_image.rows; i++)
    {
        for (j = 0; j < binary_image.cols; j++)
        {
            if (hsv.at<Vec3b>(i, j)[1] < 50 && hsv.at<Vec3b>(i, j)[2] < 85)
            {
                binary_image.at<uchar>(i, j) = 1; // Dark Gray
            }

            else
            {
                binary_image.at<uchar>(i, j) = 0;
            }
        }
    }

    nLabels = connectedComponentsWithStats(binary_image, labelImage, stats, centroids, 8, CV_32S);

    write(file, "Number of Generated Segments", nLabels);

    for (int label = 1; label < nLabels; ++label)
    {
        if (stats.at<int>(label, CC_STAT_AREA) > (gray_image.rows * gray_image.cols * size_thresh))
        {
            Vec2i center = centroids.row(label);
            bigger_segN++;

            sprintf_s(name, 1000, "Segment_No_%d", label);

            file << name;
            file << "{" << "Center Point" << center;
            file << "Size" << stats.at<int>(label, CC_STAT_AREA) << "}";

            Rect rect(stats.at<int>(label, CC_STAT_LEFT), stats.at<int>(label, CC_STAT_TOP),
                stats.at<int>(label, CC_STAT_WIDTH), stats.at<int>(label, CC_STAT_HEIGHT));

            rectangle(rect_img, rect, Scalar(0, 0, 255));
            drawMarker(rect_img, center, Scalar(0, 255, 0), 0, 10, 1, 8);
        }
    }

    write(file, "Total Number of Tresholded Segments", bigger_segN);

    ////////////////////////////////////////////////////////////////////

    bigger_segN = 0;

    for (i = 0; i < binary_image.rows; i++)
    {
        for (j = 0; j < binary_image.cols; j++)
        {
            if (hsv.at<Vec3b>(i, j)[1] < 30 && hsv.at<Vec3b>(i, j)[2] < 170 && hsv.at<Vec3b>(i, j)[2] > 85)
            {
                binary_image.at<uchar>(i, j) = 1; // Gray
            }

            else
            {
                binary_image.at<uchar>(i, j) = 0;
            }
        }
    }

    nLabels = connectedComponentsWithStats(binary_image, labelImage, stats, centroids, 8, CV_32S);

    write(file, "Number of Generated Segments", nLabels);

    for (int label = 1; label < nLabels; ++label)
    {
        if (stats.at<int>(label, CC_STAT_AREA) > (gray_image.rows * gray_image.cols * size_thresh))
        {
            Vec2i center = centroids.row(label);
            bigger_segN++;

            sprintf_s(name, 1000, "Segment_No_%d", label);

            file << name;
            file << "{" << "Center Point" << center;
            file << "Size" << stats.at<int>(label, CC_STAT_AREA) << "}";

            Rect rect(stats.at<int>(label, CC_STAT_LEFT), stats.at<int>(label, CC_STAT_TOP),
                stats.at<int>(label, CC_STAT_WIDTH), stats.at<int>(label, CC_STAT_HEIGHT));

            rectangle(rect_img, rect, Scalar(0, 0, 255));
            drawMarker(rect_img, center, Scalar(0, 255, 0), 0, 10, 1, 8);
        }
    }

    write(file, "Total Number of Tresholded Segments", bigger_segN);

    ////////////////////////////////////////////////////////////////////

    bigger_segN = 0;

    for (i = 0; i < binary_image.rows; i++)
    {
        for (j = 0; j < binary_image.cols; j++)
        {
            if (hsv.at<Vec3b>(i, j)[1] < 15 && hsv.at<Vec3b>(i, j)[2] > 170)
            {
                binary_image.at<uchar>(i, j) = 1; // Light Gray
            }

            else
            {
                binary_image.at<uchar>(i, j) = 0;
            }
        }
    }

    nLabels = connectedComponentsWithStats(binary_image, labelImage, stats, centroids, 8, CV_32S);

    write(file, "Number of Generated Segments", nLabels);

    for (int label = 1; label < nLabels; ++label)
    {
        if (stats.at<int>(label, CC_STAT_AREA) > (gray_image.rows * gray_image.cols * size_thresh))
        {
            Vec2i center = centroids.row(label);
            bigger_segN++;

            sprintf_s(name, 1000, "Segment_No_%d", label);

            file << name;
            file << "{" << "Center Point" << center;
            file << "Size" << stats.at<int>(label, CC_STAT_AREA) << "}";

            Rect rect(stats.at<int>(label, CC_STAT_LEFT), stats.at<int>(label, CC_STAT_TOP),
                stats.at<int>(label, CC_STAT_WIDTH), stats.at<int>(label, CC_STAT_HEIGHT));

            rectangle(rect_img, rect, Scalar(0, 0, 255));
            drawMarker(rect_img, center, Scalar(0, 255, 0), 0, 10, 1, 8);
        }
    }

    write(file, "Total Number of Tresholded Segments", bigger_segN);

    ////////////////////////////////////////////////////////////////////

    bigger_segN = 0;

    for (k = 0; k < 18; k = k + 3)
    {
        for (i = 0; i < binary_image.rows; i++)
        {
            for (j = 0; j < binary_image.cols; j++)
            {
                if (hsv.at<Vec3b>(i, j)[0] > abs(color_hsv[k]) / 2 - 15 && hsv.at<Vec3b>(i, j)[0] < abs(color_hsv[k]) / 2 + 15)
                {
                    binary_image.at<uchar>(i, j) = 1; // Other Colors
                }

                else
                {
                    binary_image.at<uchar>(i, j) = 0;
                }
            }
        }

        nLabels = connectedComponentsWithStats(binary_image, labelImage, stats, centroids, 8, CV_32S);

        write(file, "Number of Generated Segments", nLabels);

        for (int label = 1; label < nLabels; ++label)
        {
            if (stats.at<int>(label, CC_STAT_AREA) > (gray_image.rows * gray_image.cols * size_thresh))
            {
                Vec2i center = centroids.row(label);
                bigger_segN++;

                sprintf_s(name, 1000, "Segment_No_%d", label);

                file << name;
                file << "{" << "Center Point" << center;
                file << "Size" << stats.at<int>(label, CC_STAT_AREA) << "}";

                Rect rect(stats.at<int>(label, CC_STAT_LEFT), stats.at<int>(label, CC_STAT_TOP),
                    stats.at<int>(label, CC_STAT_WIDTH), stats.at<int>(label, CC_STAT_HEIGHT));

                    rectangle(rect_img, rect, Scalar(0, 0, 255));
                    drawMarker(rect_img, center, Scalar(0, 255, 0), 0, 10, 1, 8);
            }
        }

        write(file, "Total Number of Tresholded Segments", bigger_segN);
    }

    file.release();

    imshow("rect", rect_img);
    waitKey(0);
}

void segmentation::hsv_cc2(Mat image)
{
    int i, j, k;

    Mat hsv, gray_image;
    
    cvtColor(image, hsv, COLOR_BGR2HSV);
    cvtColor(image, gray_image, COLOR_BGR2GRAY);

    Mat binary_image = Mat::zeros(gray_image.size(), CV_8UC1);

    for (i = 0; i < binary_image.rows; i++)
    {
        for (j = 0; j < binary_image.cols; j++)
        {
            if (hsv.at<Vec3b>(i, j)[2] > 190 && hsv.at<Vec3b>(i, j)[1] < 25)
            {
                binary_image.at<uchar>(i, j) = 1; // White
            }
            else if (hsv.at<Vec3b>(i, j)[2] < 15)
            {
                binary_image.at<uchar>(i, j) = 2; // Black
            }

            else if (hsv.at<Vec3b>(i, j)[1] < 25 && hsv.at<Vec3b>(i, j)[2] < 65)
            {
                binary_image.at<uchar>(i, j) = 3; // Dark gray
            }

            else if (hsv.at<Vec3b>(i, j)[1] < 25 && hsv.at<Vec3b>(i, j)[2] < 150 && hsv.at<Vec3b>(i, j)[2] > 65)
            {
                binary_image.at<uchar>(i, j) = 4; // grey
            }

            else if (hsv.at<Vec3b>(i, j)[1] < 25 && hsv.at<Vec3b>(i, j)[2] > 150)
            {
                binary_image.at<uchar>(i, j) = 5; // light grey
            }

            else
            {
                int temp = 6;

                for (k = 0; k < 18; k = k + 3)
                {
                    if (hsv.at<Vec3b>(i, j)[0] > abs(color_hsv[k]) / 2 - 15 && hsv.at<Vec3b>(i, j)[0] < abs(color_hsv[k]) / 2 + 15)     // Check the hue to color other pixels to white
                    {
                        binary_image.at<uchar>(i, j) = temp;
                        /*
                        if (hsv.at<Vec3b>(i, j)[1] > 170 && hsv.at<Vec3b>(i, j)[2] > 170)
                        {
                            binary_image.at<uchar>(i, j) = temp + 1; // normal
                        }
                        else if (hsv.at<Vec3b>(i, j)[2] > 85 && hsv.at<Vec3b>(i, j)[2] < 170 && hsv.at<Vec3b>(i, j)[1] > 170)
                        {
                            binary_image.at<uchar>(i, j) = temp + 2;  // dark
                        }
                        else if (hsv.at<Vec3b>(i, j)[1] > 85 && hsv.at<Vec3b>(i, j)[1] < 170 && hsv.at<Vec3b>(i, j)[2] > 170)
                        {
                            binary_image.at<uchar>(i, j) = temp; // light
                        } */
                    }
                    temp++;
                }
            }
        }
    }

    Mat labelImage = Mat::zeros(gray_image.size(), CV_8UC1);

    int label = 0;

    cout << "Labelling..." << endl;

    for (i = 1; i < binary_image.rows; i++)
    {

        cout << "%" << i * 100 / binary_image.rows << endl;

        for (j = 1; j < binary_image.cols; j++)
        {

            if (binary_image.at<uchar>(i - 1, j) != binary_image.at<uchar>(i, j) && binary_image.at<uchar>(i, j - 1) != binary_image.at<uchar>(i, j))
            {
                label++;
                labelImage.at<uchar>(i, j) = label;
            }

            else if (binary_image.at<uchar>(i - 1, j) == binary_image.at<uchar>(i, j) || binary_image.at<uchar>(i, j - 1) == binary_image.at<uchar>(i, j))
            {
                if (binary_image.at<uchar>(i - 1, j) == binary_image.at<uchar>(i, j) && binary_image.at<uchar>(i, j - 1) == binary_image.at<uchar>(i, j))
                {
                    labelImage.at<uchar>(i, j) = labelImage.at<uchar>(i - 1, j);

                    for (k = 1; k < i + 1; k++)
                    {
                        for (int l = 1; l < binary_image.cols; l++)
                        {
                            if (labelImage.at<uchar>(k, l) == labelImage.at<uchar>(i, j - 1))
                            {
                                labelImage.at<uchar>(k, l) = labelImage.at<uchar>(i - 1, j);
                            }
                        }
                    }
                }
                else if (binary_image.at<uchar>(i - 1, j) == binary_image.at<uchar>(i, j))
                {
                    labelImage.at<uchar>(i, j) = labelImage.at<uchar>(i - 1, j);
                }

                else if (binary_image.at<uchar>(i, j - 1) == binary_image.at<uchar>(i, j))
                {
                    labelImage.at<uchar>(i, j) = labelImage.at<uchar>(i, j - 1);
                }
            }
        }
    }

    vector<Vec3b> colors(label);

    for (int label2 = 0; label2 < label; label2++)
    {
        colors[label2] = Vec3b((rand() & 255), (rand() & 255), (rand() & 255));
    }

    Mat dst(gray_image.size(), CV_8UC3);

    for (int r = 0; r < dst.rows; ++r) {
        for (int c = 0; c < dst.cols; ++c) {
            int label3 = labelImage.at<uchar>(r, c);
            Vec3b& pixel = dst.at<Vec3b>(r, c);
            pixel = colors[label3];
        }
    }

    imshow("Connected Components", dst);
    waitKey(0);

    cout << "Creating bounding boxes..." << endl;

    vector<int> count(label, 0);
    vector<Vec2i> center(label);

    int temp = 0;

    char name[1000];

    FileStorage file("segments2_2.txt", FileStorage::WRITE);
    write(file, "Number of Generated Segments", label);

    Mat rect_image;

    cvtColor(gray_image, rect_image, COLOR_GRAY2BGR);

    for (k = 0; k < label; k++)
    {
        int center_x = 0, center_y = 0;
        int min_x = 9999, min_y = 9999;
        int max_x = 0, max_y = 0;

        for (i = 0; i < labelImage.rows; i++)
        {
            for (j = 0; j < labelImage.cols; j++)
            {
                if (k == 0)
                {
                    int clabel = labelImage.at<uchar>(i, j);
                    count[clabel] = count[clabel] + 1;

                }

                if (k == labelImage.at<uchar>(i, j))
                {
                    center_y = center_y + i;
                    center_x = center_x + j;

                    if (i < min_y)
                    {
                        min_y = i;
                    }

                    if (j < min_x)
                    {
                        min_x = j;
                    }

                    if (i > max_y)
                    {
                        max_y = i;
                    }

                    if (j > max_x)
                    {
                        max_x = j;
                    }

                }

            }
        }
        if (count[k] != 0)
        {
            center[k] = Vec2i(center_x / count[k], center_y / count[k]);
        }

        if (count[k] > (labelImage.rows * labelImage.cols * size_thresh))
        {
            temp++;

            sprintf_s(name, 1000, "Segment_No_%d", k);

            file << name;
            file << "{" << "Center Point" << center[k];
            file << "Size" << count[k] << "}";

            Rect rect(min_x, min_y, max_y-min_y, max_x-min_x);

            rectangle(rect_image, rect, Scalar(0, 0, 255));
            drawMarker(rect_image, center[k], Scalar(0, 255, 0), 0, 10, 1, 8);
        }
    }

    write(file, "Total Number of Tresholded Segments", temp);
    file.release();

    imshow("rect", rect_image);
    cv::waitKey(0);

}