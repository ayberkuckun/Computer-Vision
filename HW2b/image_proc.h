#pragma once
#include <iostream>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

class image_proc        // Main operation class
{
public:
	void split_channels(Mat image);     // Channel Split function
    void proc_rgb(Mat image);           // RGB colorspace function
    void proc_hsi(Mat image);           // HSV colorspace function  1 2 3 4
    void intensity_op(Mat image);       // Gaussian function
    void filtering(Mat image);

private:
    string color_names[25] = { "light_red", "red","dark_red",       // Color names
        "light_green","green","dark_green",
        "light_blue","blue","dark_blue",
        "light_grey","grey", "dark_grey",
        "light_yellow","yellow","dark_yellow",
        "light_cyan","cyan","dark_cyan",
        "light_pink","pink","dark_pink",
        "orange","violet","white","black" };

    string color_names2[25] = { "light_red", "red","dark_red",       // Color names
    "light_yellow","yellow","dark_yellow",
    "light_green","green","dark_green",
    "light_cyan","cyan","dark_cyan",
    "light_blue","blue","dark_blue",
    "light_pink","pink","dark_pink",
    "orange","violet","white","black", "light_grey", "grey", "dark_grey"};

    int color_rgb[25][3] = { 255, 153, 153, 255, 0, 0,  102, 0, 0,       // RGB values for colors
        153, 255, 153, 0, 255, 0, 0, 102, 0,
        153, 153, 255, 0, 0, 255,  0, 0, 102,
        192, 192, 192, 128, 128, 128, 64, 64, 64,
        255, 255, 153, 255, 255, 0, 102, 102, 0,
        153, 255, 255, 0, 255, 255, 0, 102, 102,
        255, 153, 255, 255, 0, 255, 102, 0, 102,
        255, 128, 0, 127, 0, 255, 255, 255, 255, 0, 0, 0 };

    float color_hsv[20] = { 0.1, 0, -0.1,       // Hue values for colors
        75, 75, -75,
        127.5, 127.5, -127.5,
        187.5, 187.5, -187.5,
        240, 240, -240,
        315 , 315 , -315,
        37.5, 277.5};

	Size size = Size(720,720);  // To avoid slow processing speed a fix size decided
    const char ESC_KEY = 27;
    int treshold = 80;
    string renk;

};