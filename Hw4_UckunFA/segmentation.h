#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>

using namespace cv;
using namespace std;


class segmentation
{
public:
	void cc(Mat image);
	void hsv_cc1(Mat image);
	void hsv_cc2(Mat image);

private:

	int binary_thresh = 100;
	double size_thresh = 0.1;

	string color_names2[25] = { "light_red", "red","dark_red",       // Color names
"light_yellow","yellow","dark_yellow",
"light_green","green","dark_green",
"light_cyan","cyan","dark_cyan",
"light_blue","blue","dark_blue",
"light_pink","pink","dark_pink",
"orange","violet","white","black", "light_grey", "grey", "dark_grey" };

	double color_hsv[20] = { 0.1, 0, -0.1,       // Hue values for colors
	75, 75, -75,
	127.5, 127.5, -127.5,
	187.5, 187.5, -187.5,
	240, 240, -240,
	315 , 315 , -315,
	37.5, 277.5 };
};

