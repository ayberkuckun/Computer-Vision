#include "functions.h"

void dictionary()
{
	Mat featuresUnclustered;
	Mat input;
	Mat descriptor;

	vector<KeyPoint> keypoints;

	Ptr<SIFT> sift_detector = SIFT::create();

	vector<String> fn;

	char name[100];

	for (int i = 1; i < 6; i++)
	{
		if (i >= 4)
		{
			sprintf_s(name, 100, "images/place%d/learn/*.jpg", i);
		}

		else
		{
			sprintf_s(name, 100, "images/place%d/learn/*.jpeg", i);
		}

		glob(name, fn);

		// Image reading loop.

		for (auto f : fn)
		{
			input = imread(f, IMREAD_GRAYSCALE);

			// Detection keypoints.
			sift_detector->detect(input, keypoints);

			// Compute descriptors.
			sift_detector->compute(input, keypoints, descriptor);

			// Put the all feature descriptors in a single Mat object.
			featuresUnclustered.push_back(descriptor);
		}
	}

	// Create the BoW (or BoF) trainer
	BOWKMeansTrainer bowTrainer(10);

	// Cluster the feature vectors
	Mat dictionary = bowTrainer.cluster(featuresUnclustered);

	// Store the vocabulary
	FileStorage fs("vocabulary.yml", FileStorage::WRITE);
	fs << "vocabulary" << dictionary;
	fs.release();
}

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

void svm(int process)
{
	Mat input;
	Mat descriptor;
	Mat training_data;
	Mat training_label_data;

	Ptr<SIFT> sift_detector = SIFT::create();

	char name[100];

	vector<KeyPoint> keypoints;
	vector<String> fileN;

	float tp[9][5] = { 0 }, tn[9][5] = { 0 }, fp[9][5] = { 0 }, fn[9][5] = { 0 };
	float precision[9][5], recall[9][5], fpr[9][5];

	// Create BoF (or BoW) descriptor extractor
	BOWImgDescriptorExtractor bowDE = create_bow();

	for (int i = 1; i < 6; i++)
	{
		if (i >= 4)
		{
			sprintf_s(name, 100, "images/place%d/learn/*.jpg", i);
		}

		else
		{
			sprintf_s(name, 100, "images/place%d/learn/*.jpeg", i);
		}

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

	double svm_threshold[9] = { -1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1 };

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

		svm->trainAuto(training_data, ROW_SAMPLE, temp_label_data);

		Mat test_data;
		Mat label_data;
		Mat test_label_data;
		Mat l_array;

		if (process == 3)
		{
			for (int i = 1; i < 6; i++)
			{
				if (i >= 4)
				{
					sprintf_s(name, 100, "images/place%d/test/*.jpg", i);
				}

				else
				{
					sprintf_s(name, 100, "images/place%d/test/*.jpeg", i);
				}

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

		else if (process == 2)
		{
			test_data = training_data;
			test_label_data = temp_label_data;
		}

		test_data.convertTo(test_data, CV_32F);

		Mat margin_distances;

		svm->predict(test_data, margin_distances, cv::ml::StatModel::RAW_OUTPUT);

		Mat predicted_training_data(margin_distances.size(), CV_32S, Scalar(0));

		for (int s = 0; s < 9; s++)
		{
			for (int i = 0; i < predicted_training_data.rows; i++)
			{
				if (margin_distances.at<float>(i, 0) < svm_threshold[s])
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
		}
	}

	ofstream file;
	file.open("results.txt");

	for (int i = 0; i < 3; i++)
	{
		file << "Threshold " << svm_threshold[i] << " :" << endl;
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
		file << "Threshold " << svm_threshold[i] << " :" << endl;
		file << "	" << "Precision: ";

		for (int j = 0; j < 5; j++)
		{
			file << precision[i][j] << " ";
		}

		file << endl;
		file << "	" << "Recall: ";
		for (int j = 0; j < 5; j++)
		{
			file << recall[i][j] << " ";
		}
		file << endl;
	}

	file.close();
}