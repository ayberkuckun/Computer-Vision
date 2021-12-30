#include "image_proc.h"
#include <iostream>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

//black and white düzenle // light dark düzenle // rgb hsv renkler tekrar düzenle.

void image_proc::split_channels(Mat image)
{
    Mat rgbchannel[3];
    // The actual splitting.
    split(image, rgbchannel);

    cv::imshow("Blue/Hue", rgbchannel[0]);
    cv::waitKey(0);

    cv::imshow("Green/Saturation", rgbchannel[1]);
    cv::waitKey(0);

    cv::imshow("Red/Intensitiy", rgbchannel[2]);

    cv::waitKey(0);//Wait for a keystroke in the window
    destroyAllWindows();
}

void image_proc::proc_rgb(Mat image)
{
    resize(image, image, size);
    split_channels(image);
    Mat one_color;
    int i, j, temp=-1, error;
    while (1)                       // Main operation loop
    {
        bool name = false;
        one_color = image.clone();

        while(name == false)        // Check the operation name
        {
            if (name == false)
            {
                cout << "Specify the Color." << endl;
                cin >> renk;
            }

            if (renk == "stop")
            {
                exit(1);
            }

            for (i = 0; i < 25; i++)
            {
                if (color_names[i] == renk)
                {
                    temp = i;
                    name = true;
                    break;
                }
            }
        }

        for (i = 0; i < one_color.rows; i++)        // Paint all pixels other than specified one to black
        {
            for (j = 0; j < one_color.cols; j++)
            {
                error = abs(color_rgb[temp][2] - image.at<Vec3b>(i, j)[0]) + abs(color_rgb[temp][1] - image.at<Vec3b>(i, j)[1]) + abs(color_rgb[temp][0] - image.at<Vec3b>(i, j)[2]);
                if (error > treshold)       // Check if the image pixels in a treshold range to specified color
                {
                    if (renk == "black")
                    {
                        one_color.at<Vec3b>(i, j)[0] = 255;
                        one_color.at<Vec3b>(i, j)[1] = 255;
                        one_color.at<Vec3b>(i, j)[2] = 255;
                    }
                    else
                    {
                        one_color.at<Vec3b>(i, j)[0] = 0;
                        one_color.at<Vec3b>(i, j)[1] = 0;
                        one_color.at<Vec3b>(i, j)[2] = 0;
                    }
                }
            }
        }

        Mat last;
        cv::hconcat(one_color, image, last);
        cv::imshow(renk, last);                 // Show painted image side by side with original image

        char key;
        key = (char)cv::waitKey();//Wait for a keystroke in the window
        if (key == ESC_KEY)
        {
            break;
        }
        destroyAllWindows();
    }
}

void image_proc::proc_hsi(Mat image)
{
    Mat hsv;

    resize(image, image, size);
    cvtColor(image, hsv, COLOR_BGR2HSV);
    split_channels(hsv);

    Mat one_color;
    int i, j, temp = -1;

    while (1)                                  // Main operation loop
    {
        bool name = false;
        one_color = hsv.clone();

        while (name == false)                // Check the operation name
        {
            if (name == false)
            {
                cout << "Specify the Color." << endl;
                cin >> renk;
            }

            if (renk == "stop")
            {
                exit(1);
            }

            for (i = 0; i < 25; i++)
            {
                if (color_names2[i] == renk)
                {
                    temp = i;
                    name = true;
                    break;
                }
            }
        }

        for (i = 0; i < 720; i++)
        {
            for (j = 0; j < 720; j++)
            {
                if (renk == "white")
                {
                    if (hsv.at<Vec3b>(i, j)[2] < 255 - 10)  // Check if its intensity in range of black or white.
                    {
                        one_color.at<Vec3b>(i, j)[0] = 0;
                        one_color.at<Vec3b>(i, j)[1] = 0;
                        one_color.at<Vec3b>(i, j)[2] = 0;
                    }
                    if (hsv.at<Vec3b>(i, j)[1] > 20)
                    {
                        one_color.at<Vec3b>(i, j)[0] = 0;
                        one_color.at<Vec3b>(i, j)[1] = 0;
                        one_color.at<Vec3b>(i, j)[2] = 0;
                    }
                }
                else if (renk == "black")
                {
                    if (hsv.at<Vec3b>(i, j)[2] > 30)
                    {
                        one_color.at<Vec3b>(i, j)[0] = 0;
                        one_color.at<Vec3b>(i, j)[1] = 0;
                        one_color.at<Vec3b>(i, j)[2] = 255;
                    }
                }
                else if (renk == "dark_grey")
                {
                    if (hsv.at<Vec3b>(i, j)[1] > 50)
                    {
                        one_color.at<Vec3b>(i, j)[0] = 0;
                        one_color.at<Vec3b>(i, j)[1] = 0;
                        one_color.at<Vec3b>(i, j)[2] = 0;
                    }
                    if (hsv.at<Vec3b>(i, j)[2] > 85)
                    {
                            one_color.at<Vec3b>(i, j)[0] = 0;
                            one_color.at<Vec3b>(i, j)[1] = 0;
                            one_color.at<Vec3b>(i, j)[2] = 0;
                    }
                }
                else if (renk == "grey")
                {
                    if (hsv.at<Vec3b>(i, j)[1] > 30)
                    {
                        one_color.at<Vec3b>(i, j)[0] = 0;
                        one_color.at<Vec3b>(i, j)[1] = 0;
                        one_color.at<Vec3b>(i, j)[2] = 0;
                    }
                    if (hsv.at<Vec3b>(i, j)[2] > 170 || hsv.at<Vec3b>(i, j)[2] < 85)
                    {
                            one_color.at<Vec3b>(i, j)[0] = 0;
                            one_color.at<Vec3b>(i, j)[1] = 0;
                            one_color.at<Vec3b>(i, j)[2] = 0;
                    }
                }
                else if (renk == "light_grey")
                {
                    if (hsv.at<Vec3b>(i, j)[1] > 15)
                    {
                        one_color.at<Vec3b>(i, j)[0] = 0;
                        one_color.at<Vec3b>(i, j)[1] = 0;
                        one_color.at<Vec3b>(i, j)[2] = 0;
                    }
                    if (hsv.at<Vec3b>(i, j)[2] < 170)
                    {
                            one_color.at<Vec3b>(i, j)[0] = 0;
                            one_color.at<Vec3b>(i, j)[1] = 0;
                            one_color.at<Vec3b>(i, j)[2] = 0;
                    }
                }
                else
                {
                    if (hsv.at<Vec3b>(i, j)[0] < abs(color_hsv[temp]) / 2 - 15 || hsv.at<Vec3b>(i, j)[0] > abs(color_hsv[temp]) / 2 + 15)     // Check the hue to color other pixels to white
                    {
                        one_color.at<Vec3b>(i, j)[0] = 0;
                        one_color.at<Vec3b>(i, j)[1] = 0;
                        one_color.at<Vec3b>(i, j)[2] = 0;
                    }
                    else if (color_hsv[temp] >= 0)
                    {
                        
                        if (hsv.at<Vec3b>(i, j)[1]+20 > hsv.at<Vec3b>(i, j)[2])
                        {
                            one_color.at<Vec3b>(i, j)[0] = 0;
                            one_color.at<Vec3b>(i, j)[1] = 0;
                            one_color.at<Vec3b>(i, j)[2] = 0;
                        }
                    }
                    else if (color_hsv[temp] != abs(color_hsv[temp+2]))
                    {
                        if (hsv.at<Vec3b>(i, j)[1]-20 < hsv.at<Vec3b>(i, j)[2])
                        {
                            one_color.at<Vec3b>(i, j)[0] = 0;
                            one_color.at<Vec3b>(i, j)[1] = 0;
                            one_color.at<Vec3b>(i, j)[2] = 0;
                        }
                    }
                }
            }
        }

        Mat last;
        cvtColor(one_color, one_color, COLOR_HSV2BGR);
        cv::hconcat(one_color, image, last);
        cv::imshow(renk, last);                         // Show painted image side by side with original image

        char key;
        key = (char)cv::waitKey();//Wait for a keystroke in the window
        if (key == ESC_KEY)
        {
            break;
        }
        destroyAllWindows();
    }
}

void image_proc::intensity_op(Mat image)
{
    Mat blurred, last;

    resize(image, image, size);

    GaussianBlur(image, blurred, Size(3, 3), 0);       // Gaussian blur operation

    cv::hconcat(blurred, image, last);          // Concatenate the images
    cv::imshow("blurred", last);
    cv::waitKey();
}


