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

// camera_calibration::camera_calibration() : goodInput(false) {}

void camera_calibration::write(FileStorage& fs) const                        //Write serialization for this class
{
	fs << "{"
		<< "BoardSize_Width" << boardSize.width
		<< "BoardSize_Height" << boardSize.height
		<< "Square_Size" << squareSize
		<< "Calibrate_Pattern" << patternToUse
		<< "Calibrate_NrOfFrameToUse" << nrFrames
		<< "Calibrate_FixAspectRatio" << aspectRatio
		<< "Calibrate_AssumeZeroTangentialDistortion" << calibZeroTangentDist
		<< "Calibrate_FixPrincipalPointAtTheCenter" << calibFixPrincipalPoint

		<< "Write_DetectedFeaturePoints" << writePoints
		<< "Write_extrinsicParameters" << writeExtrinsics
		<< "Write_outputFileName" << outputFileName

		<< "Show_UndistortedImage" << showUndistorsed

		<< "Input_FlipAroundHorizontalAxis" << flipVertical
		<< "Input_Delay" << delay
		<< "Input" << input
		<< "}";
}
void camera_calibration::read(const FileNode& node)                          //Read serialization for this class
{
	node["BoardSize_Width"] >> boardSize.width;
	node["BoardSize_Height"] >> boardSize.height;
	node["Calibrate_Pattern"] >> patternToUse;
	node["Square_Size"] >> squareSize;
	node["Calibrate_NrOfFrameToUse"] >> nrFrames;
	node["Calibrate_FixAspectRatio"] >> aspectRatio;
	node["Write_DetectedFeaturePoints"] >> writePoints;
	node["Write_extrinsicParameters"] >> writeExtrinsics;
	node["Write_outputFileName"] >> outputFileName;
	node["Calibrate_AssumeZeroTangentialDistortion"] >> calibZeroTangentDist;
	node["Calibrate_FixPrincipalPointAtTheCenter"] >> calibFixPrincipalPoint;
	node["Calibrate_UseFisheyeModel"] >> useFisheye;
	node["Input_FlipAroundHorizontalAxis"] >> flipVertical;
	node["Show_UndistortedImage"] >> showUndistorsed;
	node["Input"] >> input;
	node["Input_Delay"] >> delay;
	node["Fix_K1"] >> fixK1;
	node["Fix_K2"] >> fixK2;
	node["Fix_K3"] >> fixK3;
	node["Fix_K4"] >> fixK4;
	node["Fix_K5"] >> fixK5;

	validate();
}
void camera_calibration::validate()
{
	goodInput = true;
	if (boardSize.width <= 0 || boardSize.height <= 0)
	{
		cerr << "Invalid Board size: " << boardSize.width << " " << boardSize.height << endl;
		goodInput = false;
	}
	if (squareSize <= 10e-6)
	{
		cerr << "Invalid square size " << squareSize << endl;
		goodInput = false;
	}
	if (nrFrames <= 0)
	{
		cerr << "Invalid number of frames " << nrFrames << endl;
		goodInput = false;
	}

	if (input.empty())      // Check for valid input
		inputType = INVALID;
	else
	{
		if (input[0] >= '0' && input[0] <= '9')
		{
			stringstream ss(input);
			ss >> cameraID;
			inputType = CAMERA;
		}
		else
		{
			if (isListOfImages(input) && readStringList(input, imageList))
			{
				inputType = IMAGE_LIST;
				nrFrames = (nrFrames < (int)imageList.size()) ? nrFrames : (int)imageList.size();
			}
			else
				inputType = VIDEO_FILE;
		}
		if (inputType == CAMERA)
			inputCapture.open(cameraID);
		if (inputType == VIDEO_FILE)
			inputCapture.open(input);
		if (inputType != IMAGE_LIST && !inputCapture.isOpened())
			inputType = INVALID;
	}
	if (inputType == INVALID)
	{
		cerr << " Input does not exist: " << input;
		goodInput = false;
	}

	flag = 0;
	if (calibFixPrincipalPoint) flag |= CALIB_FIX_PRINCIPAL_POINT;
	if (calibZeroTangentDist)   flag |= CALIB_ZERO_TANGENT_DIST;
	if (aspectRatio)            flag |= CALIB_FIX_ASPECT_RATIO;
	if (fixK1)                  flag |= CALIB_FIX_K1;
	if (fixK2)                  flag |= CALIB_FIX_K2;
	if (fixK3)                  flag |= CALIB_FIX_K3;
	if (fixK4)                  flag |= CALIB_FIX_K4;
	if (fixK5)                  flag |= CALIB_FIX_K5;

	if (useFisheye) {
		// the fisheye model has its own enum, so overwrite the flags
		flag = fisheye::CALIB_FIX_SKEW | fisheye::CALIB_RECOMPUTE_EXTRINSIC;
		if (fixK1)                   flag |= fisheye::CALIB_FIX_K1;
		if (fixK2)                   flag |= fisheye::CALIB_FIX_K2;
		if (fixK3)                   flag |= fisheye::CALIB_FIX_K3;
		if (fixK4)                   flag |= fisheye::CALIB_FIX_K4;
		if (calibFixPrincipalPoint) flag |= fisheye::CALIB_FIX_PRINCIPAL_POINT;
	}

	calibrationPattern = NOT_EXISTING;
	if (!patternToUse.compare("CHESSBOARD")) calibrationPattern = CHESSBOARD;
	if (!patternToUse.compare("CIRCLES_GRID")) calibrationPattern = CIRCLES_GRID;
	if (!patternToUse.compare("ASYMMETRIC_CIRCLES_GRID")) calibrationPattern = ASYMMETRIC_CIRCLES_GRID;
	if (calibrationPattern == NOT_EXISTING)
	{
		cerr << " Camera calibration mode does not exist: " << patternToUse << endl;
		goodInput = false;
	}
	atImageList = 0;

}
Mat camera_calibration::nextImage()
{
	Mat result;
	if (inputCapture.isOpened())
	{
		Mat view0;
		inputCapture >> view0;
		view0.copyTo(result);
	}
	else if (atImageList < imageList.size())
		result = imread(imageList[atImageList++], IMREAD_COLOR);

	return result;
}

bool camera_calibration::readStringList(const string& filename, vector<string>& l)
{
	l.clear();
	FileStorage fs(filename, FileStorage::READ);
	if (!fs.isOpened())
		return false;
	FileNode n = fs.getFirstTopLevelNode();
	if (n.type() != FileNode::SEQ)
		return false;
	FileNodeIterator it = n.begin(), it_end = n.end();
	for (; it != it_end; ++it)
		l.push_back((string)*it);
	return true;
}

bool camera_calibration::isListOfImages(const string& filename)
{
	string s(filename);
	// Look for file extension
	if (s.find(".xml") == string::npos && s.find(".yaml") == string::npos && s.find(".yml") == string::npos)
		return false;
	else
		return true;
}

double camera_calibration::computeReprojectionErrors(const vector<vector<Point3f> >& objectPoints,
	const vector<vector<Point2f> >& imagePoints,
	const vector<Mat>& rvecs, const vector<Mat>& tvecs,
	const Mat& cameraMatrix, const Mat& distCoeffs,
	vector<float>& perViewErrors, bool fisheye)
{
	vector<Point2f> imagePoints2;
	size_t totalPoints = 0;
	double totalErr = 0, err;
	perViewErrors.resize(objectPoints.size());

	for (size_t i = 0; i < objectPoints.size(); ++i)
	{
		if (fisheye)
		{
			fisheye::projectPoints(objectPoints[i], imagePoints2, rvecs[i], tvecs[i], cameraMatrix,
				distCoeffs);
		}
		else
		{
			projectPoints(objectPoints[i], rvecs[i], tvecs[i], cameraMatrix, distCoeffs, imagePoints2);
		}
		err = norm(imagePoints[i], imagePoints2, NORM_L2);

		size_t n = objectPoints[i].size();
		perViewErrors[i] = (float)std::sqrt(err * err / n);
		totalErr += err * err;
		totalPoints += n;
	}

	return std::sqrt(totalErr / totalPoints);
}

void camera_calibration::calcBoardCornerPositions(Size boardSize, float squareSize, vector<Point3f>& corners, Pattern patternType /*= Settings::CHESSBOARD*/)
{
	corners.clear();

	switch (patternType)
	{
	case CHESSBOARD:
	case CIRCLES_GRID:
		for (int i = 0; i < boardSize.height; ++i)
			for (int j = 0; j < boardSize.width; ++j)
				corners.push_back(Point3f(j * squareSize, i * squareSize, 0));
		break;

	case ASYMMETRIC_CIRCLES_GRID:
		for (int i = 0; i < boardSize.height; i++)
			for (int j = 0; j < boardSize.width; j++)
				corners.push_back(Point3f((2 * j + i % 2) * squareSize, i * squareSize, 0));
		break;
	default:
		break;
	}
}

bool camera_calibration::runCalibration(Size& imageSize, Mat& cameraMatrix, Mat& distCoeffs,
	vector<vector<Point2f> > imagePoints, vector<Mat>& rvecs, vector<Mat>& tvecs,
	vector<float>& reprojErrs, double& totalAvgErr)
{
	//! [fixed_aspect]
	cameraMatrix = Mat::eye(3, 3, CV_64F);
	if (flag & CALIB_FIX_ASPECT_RATIO)
		cameraMatrix.at<double>(0, 0) = aspectRatio;
	//! [fixed_aspect]
	if (useFisheye) {
		distCoeffs = Mat::zeros(4, 1, CV_64F);
	}
	else {
		distCoeffs = Mat::zeros(8, 1, CV_64F);
	}

	vector<vector<Point3f> > objectPoints(1);
	calcBoardCornerPositions(boardSize, squareSize, objectPoints[0], calibrationPattern);

	objectPoints.resize(imagePoints.size(), objectPoints[0]);

	//Find intrinsic and extrinsic camera parameters
	double rms;

	if (useFisheye) {
		Mat _rvecs, _tvecs;
		rms = fisheye::calibrate(objectPoints, imagePoints, imageSize, cameraMatrix, distCoeffs, _rvecs,
			_tvecs, flag);

		rvecs.reserve(_rvecs.rows);
		tvecs.reserve(_tvecs.rows);
		for (int i = 0; i < int(objectPoints.size()); i++) {
			rvecs.push_back(_rvecs.row(i));
			tvecs.push_back(_tvecs.row(i));
		}
	}
	else {
		rms = calibrateCamera(objectPoints, imagePoints, imageSize, cameraMatrix, distCoeffs, rvecs, tvecs,
			flag);
	}

	cout << "Re-projection error reported by calibrateCamera: " << rms << endl;

	bool ok = checkRange(cameraMatrix) && checkRange(distCoeffs);

	totalAvgErr = computeReprojectionErrors(objectPoints, imagePoints, rvecs, tvecs, cameraMatrix,
		distCoeffs, reprojErrs, useFisheye);

	return ok;
}

void camera_calibration::saveCameraParams(Size& imageSize, Mat& cameraMatrix, Mat& distCoeffs,
	const vector<Mat>& rvecs, const vector<Mat>& tvecs,
	const vector<float>& reprojErrs, const vector<vector<Point2f> >& imagePoints,
	double totalAvgErr)
{
	FileStorage fs(outputFileName, FileStorage::WRITE);

	time_t tm;
	time(&tm);
	struct tm* t2 = localtime(&tm);
	char buf[1024];
	strftime(buf, sizeof(buf), "%c", t2);

	fs << "calibration_time" << buf;

	if (!rvecs.empty() || !reprojErrs.empty())
		fs << "nr_of_frames" << (int)std::max(rvecs.size(), reprojErrs.size());
	fs << "image_width" << imageSize.width;
	fs << "image_height" << imageSize.height;
	fs << "board_width" << boardSize.width;
	fs << "board_height" << boardSize.height;
	fs << "square_size" << squareSize;

	if (flag & CALIB_FIX_ASPECT_RATIO)
		fs << "fix_aspect_ratio" << aspectRatio;

	if (flag)
	{
		std::stringstream flagsStringStream;
		if (useFisheye)
		{
			flagsStringStream << "flags:"
				<< (flag & fisheye::CALIB_FIX_SKEW ? " +fix_skew" : "")
				<< (flag & fisheye::CALIB_FIX_K1 ? " +fix_k1" : "")
				<< (flag & fisheye::CALIB_FIX_K2 ? " +fix_k2" : "")
				<< (flag & fisheye::CALIB_FIX_K3 ? " +fix_k3" : "")
				<< (flag & fisheye::CALIB_FIX_K4 ? " +fix_k4" : "")
				<< (flag & fisheye::CALIB_RECOMPUTE_EXTRINSIC ? " +recompute_extrinsic" : "");
		}
		else
		{
			flagsStringStream << "flags:"
				<< (flag & CALIB_USE_INTRINSIC_GUESS ? " +use_intrinsic_guess" : "")
				<< (flag & CALIB_FIX_ASPECT_RATIO ? " +fix_aspectRatio" : "")
				<< (flag & CALIB_FIX_PRINCIPAL_POINT ? " +fix_principal_point" : "")
				<< (flag & CALIB_ZERO_TANGENT_DIST ? " +zero_tangent_dist" : "")
				<< (flag & CALIB_FIX_K1 ? " +fix_k1" : "")
				<< (flag & CALIB_FIX_K2 ? " +fix_k2" : "")
				<< (flag & CALIB_FIX_K3 ? " +fix_k3" : "")
				<< (flag & CALIB_FIX_K4 ? " +fix_k4" : "")
				<< (flag & CALIB_FIX_K5 ? " +fix_k5" : "");
		}
		fs.writeComment(flagsStringStream.str());
	}

	fs << "flags" << flag;

	fs << "fisheye_model" << useFisheye;

	fs << "camera_matrix" << cameraMatrix;
	fs << "distortion_coefficients" << distCoeffs;

	fs << "avg_reprojection_error" << totalAvgErr;
	if (writeExtrinsics && !reprojErrs.empty())
		fs << "per_view_reprojection_errors" << Mat(reprojErrs);

	if (writeExtrinsics && !rvecs.empty() && !tvecs.empty())
	{
		CV_Assert(rvecs[0].type() == tvecs[0].type());
		Mat bigmat((int)rvecs.size(), 6, CV_MAKETYPE(rvecs[0].type(), 1));
		bool needReshapeR = rvecs[0].depth() != 1 ? true : false;
		bool needReshapeT = tvecs[0].depth() != 1 ? true : false;

		for (size_t i = 0; i < rvecs.size(); i++)
		{
			Mat r = bigmat(Range(int(i), int(i + 1)), Range(0, 3));
			Mat t = bigmat(Range(int(i), int(i + 1)), Range(3, 6));

			if (needReshapeR)
				rvecs[i].reshape(1, 1).copyTo(r);
			else
			{
				//*.t() is MatExpr (not Mat) so we can use assignment operator
				CV_Assert(rvecs[i].rows == 3 && rvecs[i].cols == 1);
				r = rvecs[i].t();
			}

			if (needReshapeT)
				tvecs[i].reshape(1, 1).copyTo(t);
			else
			{
				CV_Assert(tvecs[i].rows == 3 && tvecs[i].cols == 1);
				t = tvecs[i].t();
			}
		}
		fs.writeComment("a set of 6-tuples (rotation vector + translation vector) for each view");
		fs << "extrinsic_parameters" << bigmat;
	}

	if (writePoints && !imagePoints.empty())
	{
		Mat imagePtMat((int)imagePoints.size(), (int)imagePoints[0].size(), CV_32FC2);
		for (size_t i = 0; i < imagePoints.size(); i++)
		{
			Mat r = imagePtMat.row(int(i)).reshape(2, imagePtMat.cols);
			Mat imgpti(imagePoints[i]);
			imgpti.copyTo(r);
		}
		fs << "image_points" << imagePtMat;
	}
}

bool camera_calibration::runCalibrationAndSave(Size imageSize, Mat& cameraMatrix, Mat& distCoeffs,
	vector<vector<Point2f> > imagePoints)
{
	vector<Mat> rvecs, tvecs;
	vector<float> reprojErrs;
	double totalAvgErr = 0;

	bool ok = runCalibration(imageSize, cameraMatrix, distCoeffs, imagePoints, rvecs, tvecs, reprojErrs,
		totalAvgErr);
	cout << (ok ? "Calibration succeeded" : "Calibration failed")
		<< ". avg re projection error = " << totalAvgErr << endl;

	if (ok)
		saveCameraParams(imageSize, cameraMatrix, distCoeffs, rvecs, tvecs, reprojErrs, imagePoints,
			totalAvgErr);
	return ok;
}

// calculate calibration parameters.
void camera_calibration::calibrate_frames(Mat& cameraMatrix, Mat& distCoeffs, Size& imageSize)
{
	vector<vector<Point2f> > imagePoints;
	int mode = inputType == IMAGE_LIST ? CAPTURING : DETECTION;
	clock_t prevTimestamp = 0;
	const Scalar RED(0, 0, 255), GREEN(0, 255, 0);
	const char ESC_KEY = 27;

	//! [get_input]
	for (;;)
	{
		Mat view;
		bool blinkOutput = false;

		view = nextImage();

		//-----  If no more image, or got enough, then stop calibration and show result -------------
		if (mode == CAPTURING && imagePoints.size() >= (size_t)nrFrames)
		{
			if (runCalibrationAndSave(imageSize, cameraMatrix, distCoeffs, imagePoints))
				mode = CALIBRATED;
			else
				mode = DETECTION;
		}
		if (view.empty())          // If there are no more images stop the loop
		{
			// if calibration threshold was not reached yet, calibrate now
			if (mode != CALIBRATED && !imagePoints.empty())
				runCalibrationAndSave(imageSize, cameraMatrix, distCoeffs, imagePoints);
			break;
		}
		//! [get_input]

		imageSize = view.size();  // Format input image.
		if (flipVertical)    flip(view, view, 0);

		//! [find_pattern]
		vector<Point2f> pointBuf;

		bool found;

		int chessBoardFlags = CALIB_CB_ADAPTIVE_THRESH | CALIB_CB_NORMALIZE_IMAGE;

		if (!useFisheye) {
			// fast check erroneously fails with high distortions like fisheye
			chessBoardFlags |= CALIB_CB_FAST_CHECK;
		}

		switch (calibrationPattern) // Find feature points on the input format
		{
		case CHESSBOARD:
			found = findChessboardCorners(view, boardSize, pointBuf, chessBoardFlags);
			break;
		case CIRCLES_GRID:
			found = findCirclesGrid(view, boardSize, pointBuf);
			break;
		case ASYMMETRIC_CIRCLES_GRID:
			found = findCirclesGrid(view, boardSize, pointBuf, CALIB_CB_ASYMMETRIC_GRID);
			break;
		default:
			found = false;
			break;
		}
		//! [find_pattern]
		//! [pattern_found]
		if (found)                // If done with success,
		{
			// improve the found corners' coordinate accuracy for chessboard
			if (calibrationPattern == CHESSBOARD)
			{
				Mat viewGray;
				cvtColor(view, viewGray, COLOR_BGR2GRAY);
				cornerSubPix(viewGray, pointBuf, Size(11, 11),
					Size(-1, -1), TermCriteria(TermCriteria::EPS + TermCriteria::COUNT, 30, 0.1));
			}

			if (mode == CAPTURING &&  // For camera only take new samples after delay time
				(!inputCapture.isOpened() || clock() - prevTimestamp > delay * 1e-3 * CLOCKS_PER_SEC))
			{
				imagePoints.push_back(pointBuf);
				prevTimestamp = clock();
				blinkOutput = inputCapture.isOpened();
			}

			// Draw the corners.
			drawChessboardCorners(view, boardSize, Mat(pointBuf), found);
		}
		//! [pattern_found]
//----------------------------- Output Text ------------------------------------------------
			//! [output_text]
		string msg = (mode == CAPTURING) ? "100/100" :
			mode == CALIBRATED ? "Calibrated" : "Press 'g' to start";
		int baseLine = 0;
		Size textSize = getTextSize(msg, 1, 1, 1, &baseLine);
		Point textOrigin(view.cols - 2 * textSize.width - 10, view.rows - 2 * baseLine - 10);

		if (mode == CAPTURING)
		{
			if (showUndistorsed)
				msg = format("%d/%d Undist", (int)imagePoints.size(), nrFrames);
			else
				msg = format("%d/%d", (int)imagePoints.size(), nrFrames);
		}

		putText(view, msg, textOrigin, 1, 1, mode == CALIBRATED ? GREEN : RED);

		if (blinkOutput)
			bitwise_not(view, view);
		//! [output_text]
		//------------------------- Video capture  output  undistorted ------------------------------
		//! [output_undistorted]
		if (mode == CALIBRATED && showUndistorsed)
		{
			Mat temp = view.clone();
			if (useFisheye)
				cv::fisheye::undistortImage(temp, view, cameraMatrix, distCoeffs);
			else
				undistort(temp, view, cameraMatrix, distCoeffs);
		}
		//! [output_undistorted]
		//------------------------------ Show image and check for input commands -------------------
		//! [await_input]
		imshow("Distorted", view);
		char key = (char)waitKey(inputCapture.isOpened() ? 50 : delay);

		if (key == ESC_KEY)
			break;

		if (key == 'u' && mode == CALIBRATED)
			showUndistorsed = !showUndistorsed;

		if (inputCapture.isOpened() && key == 'g')
		{
			mode = CAPTURING;
			imagePoints.clear();
		}
		//! [await_input]
	}
	cvDestroyAllWindows();
}

// calibrate and show calibrated versions of images.
void camera_calibration::undistort_and_show_newimage(Mat& cameraMatrix, Mat& distCoeffs, Size& imageSize)
{
	// -----------------------Show the undistorted image for the image list ------------------------
	//! [show_results]
	if (inputType == IMAGE_LIST && showUndistorsed)
	{
		const char ESC_KEY = 27;
		Mat view, rview, map1, map2;

		if (useFisheye)
		{
			Mat newCamMat;
			fisheye::estimateNewCameraMatrixForUndistortRectify(cameraMatrix, distCoeffs, imageSize,
				Matx33d::eye(), newCamMat, 1);
			fisheye::initUndistortRectifyMap(cameraMatrix, distCoeffs, Matx33d::eye(), newCamMat, imageSize,
				CV_16SC2, map1, map2);
		}
		else
		{
			initUndistortRectifyMap(
				cameraMatrix, distCoeffs, Mat(),
				getOptimalNewCameraMatrix(cameraMatrix, distCoeffs, imageSize, 1, imageSize, 0), imageSize,
				CV_16SC2, map1, map2);
		}

		for (size_t i = 0; i < imageList.size(); i++)
		{
			view = imread(imageList[i], IMREAD_COLOR);
			if (view.empty())
				continue;
			remap(view, rview, map1, map2, INTER_LINEAR);
			imshow("Undistorted", rview);
			char c = (char)waitKey();
			if (c == ESC_KEY || c == 'q' || c == 'Q')
				break;
		}
	}
	//! [show_results]
}

// read parameters from a xml file.
void camera_calibration::read_param(Mat& cameraMatrix, Mat& distCoeffs, Size& imageSize)
{
	string path = "out_camera_data.xml";
	FileStorage fs(path, FileStorage::READ);
	if (!fs.isOpened())
	{
		cout << "Could not open the configuration file: \"" << path << "\"" << endl;
		exit(1);
	}
	fs["camera_matrix"] >> cameraMatrix;
	fs["distortion_coefficients"] >> distCoeffs;
	fs["image_width"] >> imageSize.width;
	fs["image_height"] >> imageSize.height;

	fs.release();
}