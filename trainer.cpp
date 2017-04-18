#include "trainer.h"

#include <opencv2/core/mat.hpp>
#include <opencv2/core/operations.hpp>
#include <opencv2/core/types_c.h>
#include <opencv2/imgproc.hpp>
#include <cassert>
#include <fstream>
#include <cmath>
#include <opencv2/highgui/highgui.hpp>

using namespace std;
using namespace cv;

Trainer::Trainer()
{
	hog = HOG();
	svm = SVM();
}

//Destructor
Trainer::~Trainer()
{
}
// Training functions

SVM Trainer::getTrainedSVM()
{
	if (svm.getSvm()->isTrained())
		return svm;
	else
		cout << "SVM not trained yet" << endl;
}

//This function calls the ROI selectioning tool to crop images from a folder.
// The second argument is the folder where the cropped images will be stored.
void Trainer::buildROISet(string filePath, string destinationFolder)
{

	//Get the list of files
	vector<String> fileNames;
	glob(filePath, fileNames);
	Mat src;
	bool abort = false;

	for (size_t i = 0; i < fileNames.size(); i++)
	{
		src = imread(fileNames[i], IMREAD_GRAYSCALE);

		//Go to next file if this one has a problem
		if (!src.data)
		{
			cout << "Cannot read " << fileNames[i] << " ... going to the next one." << endl;
			continue;
		}

		//Call the ROI selection tool
		cout << "Commencing ROI selection of file " << fileNames[i] << endl;
		vector<Mat> ROIs = getROI(src, abort, fileNames[i]);

		//If the user quitted, exit the process
		if (abort) break;

		//Otherwise, store the images in the folder
		for (int j = 0; j < ROIs.size(); j++)
		{
			//string roiNamejpg = "dataset\\roi\\" + fileName.substr(fileName.find_last_of("\\") + 1, abs(fileName.size() - (fileName.find_last_of("\\") + 1)) - 4) + "_roi_" + to_string(i) + ".jpg";
			string roiNamejpg = destinationFolder + fileNames[i].substr(fileNames[i].find_last_of("\\") + 1, abs(fileNames[i].size() - (fileNames[i].find_last_of("\\") + 1)) - 4) + "_roi_" + to_string(j) + "X.jpg";
			cout << roiNamejpg << endl;
			imwrite(roiNamejpg, ROIs[j]);
		}
	}

	cout << "ROI Set finished" << endl;
}

// Allows the user to select rectangles from the image.
vector<Mat> Trainer::getROI(Mat image, bool &abort, string fileName)
{
	// Show commands
	help();

	// Flag to determine if we finished selecting ROIs
	bool esc = false;

	// Create a window to use mouse functions 
	string window = "Select ROI(s)";
	namedWindow(window, WINDOW_AUTOSIZE);

	// Vector of rectangles
	vector<Rect> ROIs;
	Rect ROI = Rect();
	setMouseCallback(window, onMouse, &ROI);
	bool flipped = false;
	int scale = 1;

	Mat imageCopy;
	Mat flipCopy;

	int key;

	for (;;)
	{
		//Image clone to visualize ROI
		imageCopy = image.clone();
		if (flipped) 
		{
			flipCopy = imageCopy.clone();
			flip(flipCopy, imageCopy, 1);
		}

		resize(imageCopy, imageCopy, Size(imageCopy.cols * scale, imageCopy.rows * scale));
		cvtColor(imageCopy, imageCopy, COLOR_GRAY2BGR);

		//Visualize actual ROI
		rectangle(imageCopy, ROI, Scalar(0, 255, 0), 2);

		//Visualize previous Rects
		if (!ROIs.empty())
		{
			for (size_t i = 0; i < ROIs.size(); i++)
			{
				rectangle(imageCopy, ROIs[i], Scalar(0, 200, 0), 2);
			}
		}
		imshow(window, imageCopy);

		//If user presses SPACE we will save the image
		key = waitKey(20);
		switch (key)
		{
			//No input
		case -1:
			continue;

			//SPACE: Save ROI 
		case 32:
			ROIs.push_back(ROI);
			cout << "ROI saved." << endl;
			break;

			//D: Delete previous ROI
		case (int)('d') :
			if (!ROIs.empty())
			{
				ROIs.pop_back();
				cout << "Last ROI erased" << endl;
			}
			else cout << "There are no ROIs saved." << endl;
			break;

			//V: Flip image
		case (int)('v') :
			flipped = !flipped;
			ROIs.clear();
			cout << "Image flipped. All previous ROIs were deleted." << endl;
			break;

			//ESC: Go to next image
		case (int)('n'):
			esc = true;
			break;

			//Q: Quit selecting ROIs
		case (int)('q') :
			abort = true;
			break;

			//H: Lower Scale
		case (int)('h') :
			if(scale>1) scale--;
			break;

			//J: Increase scale
		case (int)('j') :
			scale++;
			break;

		}

		//If ESC was pushed finish
		if (esc) break;
	}

	// Impede any more inputs 
	setMouseCallback(window, NULL, NULL);
	destroyWindow(window);

	// Retrieve the image scale and the flip state
	imageCopy = image.clone();
	resize(imageCopy, imageCopy, Size(imageCopy.cols * scale, imageCopy.rows * scale));
	if (flipped)
	{
		flipCopy = imageCopy.clone();
		flip(flipCopy, imageCopy, 1);
	}

	//Return all the ROIs cropped
	vector<Mat> croppedImages;
	for (int i = 0; i < ROIs.size(); i++)
	{
		croppedImages.push_back(image(ROIs[i]));
	}
	return croppedImages;
}

//Overrided function to return rectangle locations instead of the images.
// This rectangles are used for FPPW.
Mat Trainer::getROI(Mat image)
{
	// Show commands
	help();

	// Flag to determine if we finished selecting ROIs
	bool esc = false;

	// Create a window to use mouse functions 
	string window = "Select ROI(s)";
	namedWindow(window, WINDOW_AUTOSIZE);

	// Vector of rectangles
	vector<Rect> ROIs;
	Rect ROI = Rect();
	setMouseCallback(window, onMouse, &ROI);

	Mat imageCopy;
	Mat flipCopy;

	int key;

	for (;;)
	{
		//Image clone to visualize ROI
		imageCopy = image.clone();
		cvtColor(imageCopy, imageCopy, COLOR_GRAY2BGR);

		//Visualize actual ROI
		rectangle(imageCopy, ROI, Scalar(0, 255, 0), 2);

		//Visualize previous Rects
		if (!ROIs.empty())
		{
			for (size_t i = 0; i < ROIs.size(); i++)
			{
				rectangle(imageCopy, ROIs[i], Scalar(0, 200, 0), 2);
			}
		}
		imshow(window, imageCopy);

		//If user presses SPACE we will save the image
		key = waitKey(20);
		switch (key)
		{
			//No input
		case -1:
			continue;

			//SPACE: Save ROI 
		case 32:
			ROIs.push_back(ROI);
			cout << "ROI saved." << endl;
			break;

			//D: Delete previous ROI
		case (int)('d') :
			if (!ROIs.empty())
			{
				ROIs.pop_back();
				cout << "Last ROI erased" << endl;
			}
			else cout << "There are no ROIs saved." << endl;
			break;

			//ESC: Go to next image
		case (int)('n') :
			esc = true;
			break;

		}

		//If ESC was pushed finish
		if (esc) break;
	}

	// Impede any more inputs 
	setMouseCallback(window, NULL, NULL);
	destroyWindow(window);

	// Store ROIs location in Mat
	Mat locationsROI = Mat::zeros(Size(4,ROIs.size()), CV_32F);
	for (int i = 0; i < ROIs.size(); i++)
	{
		locationsROI.at<float>(i, 0) = ROIs[i].x;
		locationsROI.at<float>(i, 1) = ROIs[i].y;
		locationsROI.at<float>(i, 2) = ROIs[i].width;
		locationsROI.at<float>(i, 3) = ROIs[i].height;

	}
	//cout << locationsROI << endl;
	return locationsROI;
}

//Computes the HoG descriptors of all images in a directory and stores them in a .xml file.
void Trainer::buildHOGSet(string imgDir, string setName)
{
	vector<String> fileNames;
	FileStorage fs(setName+"_hog.xml", FileStorage::WRITE);

	//Get the list of files
	glob(imgDir, fileNames);
	int j = 0;
	for (int i = 0; i < fileNames.size(); i++)
	{
	
		//If it's a PNG or a PGM
		Mat currentImage = imread(fileNames[i], CV_LOAD_IMAGE_GRAYSCALE);
		Mat currentHog = hog.getHOG(currentImage, false);
		fs << "hog_"+to_string(j++) << currentHog;
	}

	fs.release();
	cout << "HOG set " << setName << " built" << endl;
}

void Trainer::train(int pos_sample_size, int neg_sample_size, bool saveSVM)
{
	cout << "Training svm..." << endl;
	Mat features, labels = Mat(Size(1,pos_sample_size+neg_sample_size),CV_32S);
	Mat positive_sample = takeHOGSampleFromFile("positive_hog.xml",0,pos_sample_size);
	Mat negative_sample = takeHOGSampleFromFile("negative_hog.xml", 0,neg_sample_size);

	vconcat(positive_sample, negative_sample, features);
	for (int i = 0; i < features.rows; i++)
		labels.at<int>(i, 0) = i < pos_sample_size ? 1 : -1;

	svm.getSvm()->train(features, ml::ROW_SAMPLE ,labels);

	
	if(svm.getSvm()->isTrained()) cout << "SVM trained" << endl;
	else cout << "Error: SVM not trained. " << endl;

	if (saveSVM)
	{
		svm.getSvm()->save("trained_svm.xml");
		cout << "Saving trained svm to trained_svm.xml" << endl;
	}
		
}

// Perform a K-fold cross validation of the data. 
float Trainer::crossValidation(int fold_number, double c, double nu, int degree)
{
	int pos_fold_size = SAMPLE_NUMBER / fold_number;
	int neg_fold_size = SAMPLE_NUMBER * 2 / fold_number;
	//Mat features, labels = Mat(Size(1, SAMPLE_NUMBER * 2), CV_32S);
	Mat positive_sample = takeHOGSampleFromFile("positive_hog.xml", 0, SAMPLE_NUMBER);
	Mat negative_sample = takeHOGSampleFromFile("negative_hog.xml", 0, SAMPLE_NUMBER*2);
	Mat pos_label = Mat::ones(Size(1, 1), CV_32S);
	Mat neg_label = pos_label.clone();
	neg_label.at<int>(0, 0) = -1;

	float accuracy = 0.0f;

	//Try each fold as validation and all others as training set
	for (int fold = 0; fold < fold_number; fold++)
	{
		Mat training_sample = Mat::zeros(Size(positive_sample.cols, 1), positive_sample.type());
		Mat training_labels = Mat::zeros(Size(1, 1), CV_32S);
		Mat validation_sample = training_sample.clone();
		Mat validation_labels = training_labels.clone();
		//Concatenar las positivas
		for (int sample = 0; sample < SAMPLE_NUMBER; sample++)
		{
			if (sample < fold * pos_fold_size || sample >= (fold + 1)* pos_fold_size)
			{
				vconcat(training_sample, positive_sample.row(sample), training_sample);
				vconcat(training_labels, pos_label, training_labels);
			}
			else
			{
				vconcat(validation_sample, positive_sample.row(sample), validation_sample);
				vconcat(validation_labels, pos_label, validation_labels);
			}
		}
		for (int sample = 0; sample < SAMPLE_NUMBER*2; sample++)
		{
			if (sample < fold * neg_fold_size || sample >= (fold + 1)* neg_fold_size)
			{
				vconcat(training_sample, negative_sample.row(sample), training_sample);
				vconcat(training_labels, neg_label, training_labels);
			}
			else
			{
				vconcat(validation_sample, negative_sample.row(sample), validation_sample);
				vconcat(validation_labels, neg_label, validation_labels);
			}
		}

		Mat ts = training_sample.rowRange(1, training_sample.rows);
		Mat tl = training_labels.rowRange(1, training_labels.rows);
		Mat vs = validation_sample.rowRange(1, validation_sample.rows);
		Mat vl = validation_labels.rowRange(1, validation_labels.rows);
		//cout << "Rows in training sample: " << ts.rows << "\nRows in validation set: " << vs.rows << endl;
		//LINEAR KERNEL
		//svm.setParams(c, nu, degree);
		//POLY KERNEL
		svm.setParams(c, nu, degree,ml::SVM::POLY);
		svm.getSvm()->train(ts, ml::ROW_SAMPLE, tl);

		float fold_accuracy = 0.0f;
		float hitCount = 0;
		int response;
		//cout << vl << endl;
		for (int s = 0; s < vs.rows; s++)
		{
			response = svm.getSvm()->predict(vs.row(s));
			//cout << "PREDICTED: " << response << "  REAL LABEL: " << vl.at<int>(s, 0) << endl;
			if (response == vl.at<int>(s, 0))
				hitCount++;
		}
		fold_accuracy = (float)hitCount / (float)(vs.rows);
		//cout << "Fold " << fold << " accuracy: " << fold_accuracy << endl;
		accuracy += fold_accuracy;
	}

	accuracy /= (float)fold_number;

	//cout << "Accuracy for degree = "<< degree <<" C = " << c << " and Learning rate = " << nu <<" : " << accuracy << endl;

	return accuracy;

}

//Retrieves a single HoG histogram of an image.
//This function is called once per image in buildHOGSet.
Mat Trainer::takeHOGSampleFromFile(string filename, int offset, int size)
{
	FileStorage fs(filename, FileStorage::READ);
	Mat col, row;
	fs["hog_" + to_string((offset))] >> col;
	transpose(col, row);
	Mat sample = row;

	for (int i = 1; i < size; i++) {
		fs["hog_" + to_string((offset + i))] >> col;
		transpose(col, row);
		vconcat(sample, row, sample);
	}

	return sample;
}

// Function for handling mouse events
void onMouse(int event, int x, int y, int flags, void *userdata)

{
	static bool selectedObject = false;
	static Point origin;
	static Point end;

	if (selectedObject)
	{
		end = Point(x, y);
		Rect* selection = (Rect*)userdata;
		if (origin.x < end.x) selection->x = origin.x; else selection->x = end.x;
		if (origin.y < end.y) selection->y = origin.y; else selection->y = end.y;

		//selection->y = origin.y;
		selection->width = abs(origin.x - end.x);
		selection->height = abs(origin.y - end.y);

	}

	switch (event)
	{
	case EVENT_LBUTTONDOWN:
		origin = Point(x, y);
		selectedObject = true;
		break;

	case EVENT_LBUTTONUP:
		selectedObject = false;
		break;

	}

}

void help()
{
	cout << "ROI capture initialized: Press SPACE to save a ROI or D to delete previous ROI." << endl;
}

// Tries a different combination of parameters.
// The set that performs best (in terms of accuracy) is selected.
void Trainer::setOptimalParameters(bool saveParams) 
{
	double c = 0.00001;
	double nu = 0.1;
	float accuracy, bestAccuracy = 0.0f;
	double best_c = 0.0f, best_nu = 0.0f;
	int best_degree = 1;
	int degree = 1;
	for (int degree = 1; degree < 4; degree++)
	{
		for (double c = 0.1; c < 10; c+=0.1)
		{
			nu = 0.1;
			//for (int j = 0; j < 10; j++)
			//{
				accuracy = crossValidation(10, c, nu, degree);
				if (accuracy > bestAccuracy)
				{
					best_c = c;
					best_nu = nu;
					best_degree = degree;
					bestAccuracy = accuracy;
				}
				//nu += 0.1;
			//}
			c *= 10;
		}
	}

	cout << "BEST PARAMETERS are c =  " << best_c << "  nu = " << best_nu << " degree = " << best_degree << endl;
	cout << "Accuracy for best parameters: " << bestAccuracy << endl;
	//svm.setParams(best_c, best_nu, best_degree);

	if (saveParams)
	{
			svm.setParams(best_c, best_nu, best_degree, ml::SVM::POLY);
	}
}