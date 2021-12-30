#include "dummy.h"

void dummy::dictionary()
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
		sprintf_s(name, 100, "images/%d/Train/*.jpg", i);

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
	BOWKMeansTrainer bowTrainer(5);

	// Cluster the feature vectors
	Mat dictionary = bowTrainer.cluster(featuresUnclustered);

	// Store the vocabulary
	FileStorage fs("vocabulary.yml", FileStorage::WRITE);
	fs << "vocabulary" << dictionary;
	fs.release();
}

void dummy::bow(string mod)
{
	Mat input;
	Mat descriptor, d_descriptor;

	vector<KeyPoint> keypoints;

	char name[100];
	char name2[100];
	
	Ptr<SIFT> sift_detector = SIFT::create();

	// Read the vocabulary    
	Mat dictionary;
	FileStorage fs("vocabulary.yml", FileStorage::READ);
	fs["vocabulary"] >> dictionary;
	fs.release();

	vector<String> fn;

	// Create a FLANN based matcher
	Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create(DescriptorMatcher::FLANNBASED);

	// Create Sift descriptor extractor
	Ptr<DescriptorExtractor> extractor = SIFT::create();

	// Create BoF (or BoW) descriptor extractor
	BOWImgDescriptorExtractor bowDE(extractor, matcher);

	// Set the dictionary
	bowDE.setVocabulary(dictionary);

	//double class_mean[5][5];
	Mat class_mean(5, 5, CV_32F);
	class_mean.setTo(0);

	int imNo;

	//ofstream file;
	//file.open("class_measn.txt");
	FileStorage fs2("class_means.yml", FileStorage::WRITE);

	for (int i = 1; i < 6; i++)
	{
		sprintf_s(name, 100, "images/%d/Train/*.jpg", i);

		glob(name, fn);

		imNo = 0;

		//memset(class_mean, 0, sizeof(class_mean));
		//class_mean.setTo(0);

		// Image reading loop.

		for (auto f : fn)
		{
			input = imread(f, IMREAD_GRAYSCALE);

			// Detect key points.
			sift_detector->detect(input, keypoints);

			// Compute descriptor vectors.
			bowDE.compute(input, keypoints, descriptor);
			
			descriptor.convertTo(d_descriptor, CV_32F);

			for (int j = 0; j < 5; j++)
			{
				class_mean.at<float>(i - 1, j) += d_descriptor.at<float>(0, j);
			}

			imNo++;
		}

		for (int j = 0; j < 5; j++)
		{
			class_mean.at<float>(i - 1, j) = class_mean.at<float>(i - 1, j) / double(imNo);
		}

		sprintf_s(name2, 100, "class_%d_mean", i);
		fs2 << name2 << class_mean.row(i-1);

		//file << name2;

		//for (int j = 0; j < 5; j++)
		//{
			//file << class_mean[i-1][j] << " ";
			//cout << class_mean[i-1][j] << " ";
		//}
		//cout << endl;
		//file << endl;
	}

	fs2.release();
	//file.close();
	
	double similarity[5][5] = { 0 };

	Mat class_descriptors(imNo, 5, CV_32F);

	for (int i = 1; i < 6; i++)
	{
		imNo = 0;

		if (mod == "train")
		{
			sprintf_s(name, 100, "images/%d/Train/*.jpg", i);
		}

		else if (mod == "test")
		{
			sprintf_s(name, 100, "images/%d/Test/*.jpg", i);
		}

		glob(name, fn);

		//memset(similarity, 0, sizeof(similarity));

		// Image reading loop.

		for (auto f : fn)
		{
			input = imread(f, IMREAD_GRAYSCALE);

			// Detect key points.
			sift_detector->detect(input, keypoints);

			// Compute descriptor vectors.
			bowDE.compute(input, keypoints, descriptor);

			descriptor.convertTo(d_descriptor, CV_32F);

			for (int j = 0; j < 5; j++)
			{
				class_descriptors.at<float>(imNo, j) = d_descriptor.at<float>(0, j);
			}

			imNo++;
		}

		for (int j = 0; j < 5; j++)
		{
			for (int k = 0; k < imNo ; k++)
			{
				similarity[i - 1][j] += norm(class_mean.row(j), class_descriptors.row(k), NORM_L2);
			}
		}

		for (int j = 0; j < 5; j++)
		{
			similarity[i - 1][j] = similarity[i - 1][j] / double(imNo);
		}

	}

	ofstream matf;

	if (mod == "train")
	{
		matf.open("train_similarity_matrix.txt");
	}

	else if (mod == "test")
	{
		matf.open("test_similarity_matrix.txt");
	}
	
	for (int i = 0; i < 5; i++)
	{
		for (int j = 0; j < 5; j++)
		{
			matf << similarity[i][j] << " ";
		}

		matf << endl;
	}

	matf.close(); 
}