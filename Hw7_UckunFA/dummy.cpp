#include "dummy.h"

BOWImgDescriptorExtractor create_bow()
{
	// Read the vocabulary    
	Mat dictionary;
	FileStorage fs("vocabulary.yml", FileStorage::READ);
	fs["vocabulary"] >> dictionary;
	fs.release();

	// Create a FLANN based matcher
	Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create(DescriptorMatcher::FLANNBASED);

	// Create Sift descriptor extractor
	Ptr<DescriptorExtractor> extractor = SIFT::create();

	// Create BoF (or BoW) descriptor extractor
	BOWImgDescriptorExtractor bowDE(extractor, matcher);

	// Set the dictionary
	bowDE.setVocabulary(dictionary);

	return bowDE;
};

Ptr<SVM> svm_create()
{
	Ptr<SVM> svm = SVM::create();
	svm->setType(SVM::C_SVC);
	svm->setKernel(SVM::CHI2);
	svm->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, 1000, 1e-6));

	return svm;
};

void svm(int question)
{
	Mat input;
	Mat descriptor;
	Mat training_data;
	Mat training_label_data;

	Ptr<SIFT> sift_detector = SIFT::create();

	char name[100];

	vector<KeyPoint> keypoints;
	vector<String> fileN;

	float tp[3][5] = { 0 }, tn[3][5] = { 0 }, fp[3][5] = { 0 }, fn[3][5] = { 0 };
	float precision[3][5], recall[3][5], fpr[3][5];

	// Create BoF (or BoW) descriptor extractor
	BOWImgDescriptorExtractor bowDE = create_bow();

	for (int i = 1; i < 6; i++)
	{
		sprintf_s(name, 100, "images/%d/Train/*.jpg", i);

		glob(name, fileN);

		// Image reading loop.

		for (auto f : fileN)
		{
			input = imread(f, IMREAD_GRAYSCALE);

			// Detect key points.
			sift_detector->detect(input, keypoints);

			// Compute descriptor vectors.
			bowDE.compute(input, keypoints, descriptor);

			training_data.push_back(descriptor);
			
			training_label_data.push_back(i);
		}
	}

	training_data.convertTo(training_data, CV_32F);

	// Train the SVM
	Ptr<SVM> svm = svm_create();

	for (int k = 0; k < 5; k++)
	{
		Mat temp_label_data;

		for (int l = 0; l < training_label_data.rows; l++)
		{
			if (training_label_data.at<int>(l, 0) == k + 1)
			{
				temp_label_data.push_back(1);
			}

			else
			{
				temp_label_data.push_back(-1);
			}
		}

		//svm->setP(1);
//Ptr<ParamGrid> threshold = ParamGrid::create(0.01, 1, 1.5);
//svm->train(training_data, ROW_SAMPLE, training_label_data);
//svm->trainAuto(training_data, ROW_SAMPLE, training_label_data, 10,
	//SVM::getDefaultGridPtr(SVM::C), SVM::getDefaultGridPtr(SVM::GAMMA),
	//threshold);

		svm->trainAuto(training_data, ROW_SAMPLE, temp_label_data);

		/*
		Mat alpha, svidx;
		double rho[10];

		for (int i = 0; i < 10; i++)
		{
			rho[i] = svm->getDecisionFunction(i, alpha, svidx);
		} */

		//double p = svm->getP();

		Mat test_data;
		Mat label_data;
		Mat test_label_data;
		Mat l_array;

		if (question == 2)
		{
			for (int i = 1; i < 6; i++)
			{
				sprintf_s(name, 100, "images/%d/Test/*.jpg", i);

				glob(name, fileN);

				// Image reading loop.

				for (auto f : fileN)
				{
					input = imread(f, IMREAD_GRAYSCALE);

					// Detect key points.
					sift_detector->detect(input, keypoints);

					// Compute descriptor vectors.
					bowDE.compute(input, keypoints, descriptor);

					test_data.push_back(descriptor);

					label_data.push_back(i);
				}
			}

			for (int l = 0; l < label_data.rows; l++)
			{
				if (label_data.at<int>(l, 0) == k + 1)
				{
					test_label_data.push_back(1);
				}

				else
				{
					test_label_data.push_back(-1);
				}
			}
		}	

		else if(question == 1)
		{
			test_data = training_data;
			test_label_data = temp_label_data;
		}

		else if (question == 3)
		{
			sprintf_s(name, 100, "images/6/Test/*.jpg");

			glob(name, fileN);

			// Image reading loop.

			for (auto f : fileN)
			{
				input = imread(f, IMREAD_GRAYSCALE);

				// Detect key points.
				sift_detector->detect(input, keypoints);

				// Compute descriptor vectors.
				bowDE.compute(input, keypoints, descriptor);

				test_data.push_back(descriptor);
			}

			Mat l_array(test_data.rows, 1, CV_32S, Scalar(-1));
			test_label_data = l_array;
		}

		test_data.convertTo(test_data, CV_32F);


		Mat margin_distances;

		svm->predict(test_data, margin_distances, cv::ml::StatModel::RAW_OUTPUT);

		Mat predicted_training_data(margin_distances.size(), CV_32S, Scalar(0));

		double threshold[3] = { 0, 0.5, 1 };

		for (int s = 0; s < 3; s++)
		{
			for (int i = 0; i < predicted_training_data.rows; i++)
			{
				if (margin_distances.at<float>(i, 0) < threshold[s])
				{
					predicted_training_data.at<int>(i, 0) = 1;
				}

				else
				{
					predicted_training_data.at<int>(i, 0) = -1;
				}
			}

			for (int j = 0; j < predicted_training_data.rows; j++)
			{
				if (predicted_training_data.at<int>(j, 0) == 1)
				{
					if (predicted_training_data.at<int>(j, 0) == test_label_data.at<int>(j, 0))
					{
						tp[s][k]++;
					}

					else
					{
						fp[s][k]++;
					}
				}

				else if (predicted_training_data.at<int>(j, 0) == -1)
				{
					if (test_label_data.at<int>(j, 0) == -1)
					{
						tn[s][k]++;
					}

					else
					{
						fn[s][k]++;
					}
				}
			}

			precision[s][k] = tp[s][k] / (tp[s][k] + fp[s][k]);
			recall[s][k] = tp[s][k] / (tp[s][k] + fn[s][k]);
			fpr[s][k] = fp[s][k] / (fp[s][k] + tn[s][k]);
		}
	}

	float threshold_mean_precision[3] = { 0 };
	float threshold_mean_recall[3] = { 0 };
	float class_mean_precision[5] = { 0 };
	float class_mean_recall[5] = { 0 };
	float threshold_mean_fpr[3] = { 0 };
	float class_mean_fpr[5] = { 0 };

	for (int i = 0; i < 3; i++)
	{
		for (int j = 0; j < 5; j++)
		{
			threshold_mean_precision[i] += precision[i][j];
			threshold_mean_recall[i] += recall[i][j];
			threshold_mean_fpr[i] += fpr[i][j];
		}
	}

	for (int j = 0; j < 3; j++)
	{
		threshold_mean_precision[j] /= 5;
		threshold_mean_recall[j] /= 5;
		threshold_mean_fpr[j] /= 5;
	}

	for (int i = 0; i < 5; i++)
	{
		for (int j = 0; j < 3; j++)
		{
			class_mean_precision[i] += precision[j][i];
			class_mean_recall[i] += recall[j][i];
			class_mean_fpr[i] += fpr[j][i];
		}
	}

	for (int j = 0; j < 5; j++)
	{
		class_mean_precision[j] /= 3;
		class_mean_recall[j] /= 3;
		class_mean_fpr[j] /= 3;
	}

	ofstream file;
	file.open("results.txt");

	for (int i = 0; i < 3; i++)
	{
		file << "Threshold " << i + 1 << ":" << endl;
		file << "	" << "TP: ";

		for (int j = 0; j < 5; j++)
		{
			file << tp[i][j] << " ";
		}

		file << endl;
		file << "	" << "TN: ";

		for (int j = 0; j < 5; j++)
		{
			file << tn[i][j] << " ";
		}

		file << endl;
		file << "	" << "FP: ";

		for (int j = 0; j < 5; j++)
		{
			file << fp[i][j] << " ";
		}

		file << endl;
		file << "	" << "FN: ";

		for (int j = 0; j < 5; j++)
		{
			file << fn[i][j] << " ";
		}

		file << endl;
	}

	file << endl;
	file.precision(2);

	for (int i = 0; i < 3; i++)
	{
		file << "Threshold " << i + 1 << ":" << endl;
		file << "	" << "Precision: " << threshold_mean_precision[i] << endl;
		file << "	" << "Recall: " << threshold_mean_recall[i] << endl;
		file << "	" << "FPR: " << threshold_mean_fpr[i] << endl;
	}

	file << endl;

	for (int i = 0; i < 5; i++)
	{
		file << "Class " << i + 1 << ":" << endl;
		file << "	" << "Precision: " << class_mean_precision[i] << endl;
		file << "	" << "Recall: " << class_mean_recall[i] << endl;
		file << "	" << "FPR: " << class_mean_fpr[i] << endl;
	}

	file.close();
}
