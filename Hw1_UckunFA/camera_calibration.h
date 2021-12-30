#pragma once

#include <iostream>
#include <sstream>
#include <string>
#include <ctime>
#include <cstdio>

#include <opencv2/core.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>


using namespace cv;
using namespace std;

class camera_calibration
{
public:
    //camera_calibration() : goodInput(false) {}
    enum Pattern { NOT_EXISTING, CHESSBOARD, CIRCLES_GRID, ASYMMETRIC_CIRCLES_GRID };
    enum InputType { INVALID, CAMERA, VIDEO_FILE, IMAGE_LIST };
    enum { DETECTION = 0, CAPTURING = 1, CALIBRATED = 2 };

    void write(FileStorage& fs) const;                        //Write serialization for this class
    void read(const FileNode& node);                         //Read serialization for this class
    void validate();

    Mat nextImage();

    bool readStringList(const string& filename, vector<string>& l);
    bool isListOfImages(const string& filename);

    double computeReprojectionErrors(const vector<vector<Point3f> >& objectPoints,
        const vector<vector<Point2f> >& imagePoints,
        const vector<Mat>& rvecs, const vector<Mat>& tvecs,
        const Mat& cameraMatrix, const Mat& distCoeffs,
        vector<float>& perViewErrors, bool fisheye);
    void calcBoardCornerPositions(Size boardSize, float squareSize, vector<Point3f>& corners,
        Pattern patternType /*= Settings::CHESSBOARD*/);
    bool runCalibration(Size& imageSize, Mat& cameraMatrix, Mat& distCoeffs,
        vector<vector<Point2f> > imagePoints, vector<Mat>& rvecs, vector<Mat>& tvecs,
        vector<float>& reprojErrs, double& totalAvgErr);
    void saveCameraParams(Size& imageSize, Mat& cameraMatrix, Mat& distCoeffs,
        const vector<Mat>& rvecs, const vector<Mat>& tvecs,
        const vector<float>& reprojErrs, const vector<vector<Point2f> >& imagePoints,
        double totalAvgErr);
    bool runCalibrationAndSave(Size imageSize, Mat& cameraMatrix, Mat& distCoeffs,
        vector<vector<Point2f> > imagePoints);

    void calibrate_frames(Mat& cameraMatrix, Mat& distCoeffs, Size& imageSize);
    void undistort_and_show_newimage(Mat& cameraMatrix, Mat& distCoeffs, Size& imageSize);
    void read_param(Mat& cameraMatrix, Mat& distCoeffs, Size& imageSize);
 
public:
    Size boardSize;              // The size of the board -> Number of items by width and height
    Pattern calibrationPattern;  // One of the Chessboard, circles, or asymmetric circle pattern
    float squareSize;            // The size of a square in your defined unit (point, millimeter,etc).
    int nrFrames;                // The number of frames to use from the input for calibration
    float aspectRatio;           // The aspect ratio
    int delay;                   // In case of a video input
    bool writePoints;            // Write detected feature points
    bool writeExtrinsics;        // Write extrinsic parameters
    bool calibZeroTangentDist;   // Assume zero tangential distortion
    bool calibFixPrincipalPoint; // Fix the principal point at the center
    bool flipVertical;           // Flip the captured images around the horizontal axis
    string outputFileName;       // The name of the file where to write
    bool showUndistorsed;        // Show undistorted images after calibration
    string input;                // The input ->
    bool useFisheye;             // use fisheye camera model for calibration
    bool fixK1;                  // fix K1 distortion coefficient
    bool fixK2;                  // fix K2 distortion coefficient
    bool fixK3;                  // fix K3 distortion coefficient
    bool fixK4;                  // fix K4 distortion coefficient
    bool fixK5;                  // fix K5 distortion coefficient

    int cameraID;
    vector<string> imageList;
    size_t atImageList;
    VideoCapture inputCapture;
    InputType inputType;
    bool goodInput;
    int flag;

private:
    string patternToUse;
};

