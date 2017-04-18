#include "performance.h"
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

// Writes the PR curve in a .XML file by testing, for a number of training images, 
// different values of C.
void writeAUC(int positive_sample, int negative_sample)
{
	Tester ts = Tester(); 
	Trainer tr = Trainer();

	Mat AUC = Mat::zeros(Size(3, 1), CV_32F);

	for (double c = 0.1; c < 1; c += 0.05)
	{
		Mat PR = Mat::zeros(Size(3, 1), CV_32F);
		PR.at<float>(0, 2) = (float)c;
		cout << "For C =" << (float)c << endl;
		tr.svm.setParams(c, 0.1, 1, ml::SVM::LINEAR);
		tr.train(positive_sample,negative_sample);
		ts.setSVM(tr.svm);
		ts.test("dataset\\testing\\",true,PR);

		vconcat(AUC, PR.clone(), AUC);
	}

	cout << AUC << endl;

	FileStorage fs("AUC.xml", FileStorage::WRITE);
	fs << "AUC" << AUC;
	fs.release();
}

void getObjectLocations(string fileDirectory)
{
	vector<String> fileNames;
	FileStorage fs(SHEEP_LOCATIONS_FILE, FileStorage::WRITE);

	//Declare trainer to access ROI tool
	Trainer tr = Trainer();

	//Get the list of files
	glob(fileDirectory, fileNames);

	//Get their bounding boxes and write their locations
	for (int i = 0; i < fileNames.size(); i++)
	{
		Mat roiLocations;
		Mat image = imread(fileNames[i], 0);
		roiLocations = tr.getROI(image);
		fs << "s"+to_string(i+1) << roiLocations;
	}

	fs.release();
}

//// Gets the false po
void getFPPW(string fileDirectory, bool show)
{
	FileStorage fs(SHEEP_LOCATIONS_FILE, FileStorage::READ);

	// Declare Tester object and load SVM parameters
	Tester tester = Tester();
	tester.loadTSVM("trained_svm.xml");

	//Get the list of testing images
	vector<String> fileNames;
	glob(fileDirectory, fileNames);

	//Number of true sheeps
	int true_sheep = 0;
	int real_sheep = 0;
	int false_sheep = 0;
	int found_sheep = 0;

	for (int i = 0; i < fileNames.size(); i++)
	{
		// Get list of matches
		Mat image = imread(fileNames[i], 0);

		vector<Match> nms_matches = tester.getPositiveMatches(image, false);
		vector<Match> matches;

		if (nms_matches.size() > 0) 
		{
			matches = tester.applyNMS(nms_matches);
			cout << matches.size() << " matches detected in " << fileNames[i] << endl;
		}
		else
		{
			cout << "No matches detected in " << fileNames[i] << endl;
			continue;
		}

		// Get sheep locations from files
		Mat sheepLocations; 
		fs["s"+to_string(i+1)] >> sheepLocations;
		real_sheep += sheepLocations.rows;

		for (int j = 0; j < matches.size(); j++)
		{
			//Flag to sheck if the bounding box has found a sheep assigned
			bool sheep_assigned = false;
			// We check if any of bounding have a IoU of 70% or more
			for (int k = 0; k < sheepLocations.rows; k++)
			{

				Rect R1 = Rect(matches[j].x, matches[j].y, WINDOW_WIDTH/ matches[j].scale, WINDOW_HEIGHT / matches[j].scale);
				Rect R2 = Rect(sheepLocations.at<float>(k,0), sheepLocations.at<float>(k, 1), sheepLocations.at<float>(k, 2), sheepLocations.at<float>(k, 3));
				
				// Intersection
				Rect R3 = R1 & R2;
				float area_of_intersection = R3.area();
				// Union
				float area_of_union = R1.area()+R2.area()-R3.area();

				if (sheepLocations.at<float>(k, 3) != 0.0f || sheepLocations.at<float>(k, 4) != 0.0f)
				{

					// If the area of intersection is above 
					if (area_of_intersection > 0.0f && area_of_intersection / area_of_union > IOU_THRESHOLD)
					{
						true_sheep++;
						sheep_assigned = true;
						break;
					}

				}

				//Show results
				if (show)
				{
					Mat imageCopy = image.clone();
					cvtColor(imageCopy, imageCopy, COLOR_GRAY2BGR);
					rectangle(imageCopy, R1, Scalar(0, 255, 0), 3);
					rectangle(imageCopy, R2, Scalar(0, 0, 255), 3);
					imshow("AREA", imageCopy);
					waitKey(0);
				}

				//If this window has not found any sheep, then it is considered that is a false positive
				if(!sheep_assigned)
				{
					false_sheep++;
				}

			}
			if (sheep_assigned)
			{
				found_sheep++;
			}

		}
	}

	//Ratio of accuracy
	float ratio = (float)true_sheep / (float)(true_sheep + false_sheep);
	cout << "False sheep " << false_sheep << " True sheep " << true_sheep << "RATIO" << ratio << endl;
	cout << "There were " << real_sheep << "of which " << found_sheep << "were found" << endl;
}

//Gets mean and standard deviation of height/width ratio of positive samples
void getAspectRatio()
{
	vector<String> fileNames;
	glob("dataset//roi", fileNames);
	Mat ratios = Mat::zeros(Size(1, fileNames.size()), CV_32F);
	for (int i = 0; i < fileNames.size(); i++)
	{
		Mat img = imread(fileNames[i], 0);
		ratios.at<float>(i, 0) = (float)img.cols / (float)img.rows;
	}
	Mat standard_deviation;
	Mat mean;
	meanStdDev(ratios, Mat(), standard_deviation);
	cout << standard_deviation << endl;
}