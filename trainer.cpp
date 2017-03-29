#include "trainer.h"

#include <opencv2/core/mat.hpp>
#include <opencv2/core/operations.hpp>
#include <opencv2/core/types_c.h>
#include <opencv2/imgproc.hpp>
#include <cassert>
#include <iostream>
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

void Trainer::buildROISet(string filePath)
{
	vector<String> fileNames;

	//Get the list of files
	glob(filePath, fileNames);
	Mat src;

	for (size_t i = 0; i < fileNames.size(); i++)
	{
		src = imread(fileNames[i], IMREAD_GRAYSCALE);

		//Go to next file if this one has a problem
		if (!src.data)
		{
			cout << "Cannot read " << fileNames[i] << " ... going to the next one." << endl;
			continue;
		}

		//
		bool abort;
		cout << "Commencing ROI selection of file " << fileNames[i] << endl;
		vector<Mat> ROI = getROI(src, abort, fileNames[i]);
	}


}

// Allows the user to select rectangles from the image
// and calculate their histograms.
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

		case (int)('h') :
			if(scale>1) scale--;
			break;

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

	imageCopy = image.clone();
	resize(imageCopy, imageCopy, Size(imageCopy.cols * scale, imageCopy.rows * scale));
	if (flipped)
	{
		flipCopy = imageCopy.clone();
		flip(flipCopy, imageCopy, 1);
	}

	for (int i = 0; i < ROIs.size(); i++)
	{
		//FIX: SUBSTRING NAME
		string roiName = "dataset\\roi_n\\" + fileName.substr(fileName.find_last_of("\\") + 1, abs(fileName.size() - (fileName.find_last_of("\\") + 1)) - 4) + "_roi_" + to_string(i) + ".pgm";
		string roiNamejpg = "dataset\\roi_n\\" + fileName.substr(fileName.find_last_of("\\") + 1, abs(fileName.size() - (fileName.find_last_of("\\") + 1)) - 4) + "_roi_" + to_string(i) + ".jpg";

		cout << roiName << endl;

		imwrite(roiName, imageCopy(ROIs[i]));
		imwrite(roiNamejpg, imageCopy(ROIs[i]));
	}
}

void Trainer::buildHOGSet(string imgDir, string setName)
{
	vector<String> fileNames;
	FileStorage fs(setName+"_hog.xml", FileStorage::WRITE);

	//Get the list of files
	glob(imgDir, fileNames);
	int j = 0;
	for (int i = 0; i < fileNames.size(); i++)
	{
		if (fileNames[i].find(".jpg") != std::string::npos)
			continue;
		//If it's a PNG or a PGM
		Mat currentImage = imread(fileNames[i], CV_LOAD_IMAGE_GRAYSCALE);
		Mat currentHog = hog.getHOG(currentImage);
		fs << "hog_"+to_string(j++) << currentHog;
	}

	fs.release();
}

void Trainer::train(int sample_size)
{
	Mat features, labels = Mat(Size(1,sample_size*2),CV_32S);
	Mat positive_sample = takeHOGSampleFromFile("positive_hog.xml",10,sample_size);
	Mat negative_sample = takeHOGSampleFromFile("negative_hog.xml", 20,sample_size);

	vconcat(positive_sample, negative_sample, features);
	for (int i = 0; i < sample_size * 2; i++)
		labels.at<int>(i, 0) = i < sample_size ? 1 : -1;

	cout << "FEATURES: type " << features.type() << " SIZE " << features.cols << " " << features.rows << endl;
	cout << "FEATURES: type " << labels.type() << " SIZE " << labels.cols << " " << labels.rows << endl;
	cout << labels << endl;

	svm.setParams(ml::SVM::C_SVC, ml::SVM::POLY, 3);
	svm.getSvm()->train(features, ml::ROW_SAMPLE ,labels);

	svm.getSvm()->save("trained_svm.xml");
}

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