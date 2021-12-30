/* EE576 MACHINE VISION HW1
Fehmi Ayberk Uçkun
2015401009
*/

// Add all required libraries and header files.

#include "camera_calibration.h"

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

// Settings read function.
camera_calibration read_settings(int argc, char* argv[])
{
    //! [file_read]

    camera_calibration s;
    const string inputSettingsFile = argc > 1 ? argv[1] : "default.xml";
    FileStorage fs(inputSettingsFile, FileStorage::READ); // Read the settings
    if (!fs.isOpened())
    {
        cout << "Could not open the configuration file: \"" << inputSettingsFile << "\"" << endl;
        exit(1);
    }
    fs["Settings"] >> s;
    fs.release();                                         // close Settings file
    //! [file_read]

    if (!s.goodInput)
    {
        cout << "Invalid input detected. Application stopping. " << endl;
        exit(1);
    }

    return s;

    //FileStorage fout("settings.yml", FileStorage::WRITE); // write config as YAML
    //fout << "Settings" << s;
}

static inline void read(const FileNode& node, camera_calibration& x, const camera_calibration& default_value = camera_calibration())
{
    if (node.empty())
        x = default_value;
    else
        x.read(node);
}

int main(int argc, char* argv[])
{
    // read command line argument which is a configuration file in XML format.
    camera_calibration s = read_settings(argc, argv);

    // Variable initialization.
    Mat cameraMatrix, distCoeffs;
    Size imageSize;

    // Calculate calibration parameters using an image list.
    s.calibrate_frames(cameraMatrix, distCoeffs, imageSize);

    // You can uncomment the below function and comment the above function line after saving the calibration parameters.
    // This way, after calibrating your camera, you dont need to do it again.
    
    //s.read_param(cameraMatrix, distCoeffs, imageSize);

    // calibrate and show calibrated versions of the images.
    s.undistort_and_show_newimage(cameraMatrix, distCoeffs, imageSize);

    return 0;
}