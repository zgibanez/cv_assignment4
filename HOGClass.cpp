#include "HOGClass.h"

#include <opencv2/core/mat.hpp>
#include <opencv2/core/operations.hpp>
#include <opencv2/core/types_c.h>
#include <opencv2/imgproc.hpp>
#include <cassert>
#include <iostream>
#include <fstream>
#include <cmath>
#include <opencv2/highgui/highgui.hpp>

//Drawing
#include <opencv2/video/tracking.hpp>



using namespace std;
using namespace cv;

HOG::HOG(void)
{
	setBinValues();
	setCellDimensions();
}

//Destructor
HOG::~HOG()
{
}

//Sets the boundaries of the bin values (0-20,20-40... etc) according to the
//histogram characteristics
void HOG::setBinValues()
{
	//Number of degrees to divide in number of bins
	float totalDegrees;
	if (HISTOGRAMS_360) totalDegrees = 360.0; else totalDegrees = 180.0;
	float degreesPerBin = totalDegrees / BIN_NUMBER;

	Mat binLimits = Mat::zeros(Size(1, BIN_NUMBER), CV_32F);

	for (int i = 0; i < binLimits.rows; i++)
	{
		binLimits.at<float>(0, i) = degreesPerBin*i;
	}

	cout << "Bin values are: " << binLimits << endl;
	binValues = binLimits;
	binSize = degreesPerBin;
}

// Set dimensions in pixels of the cells given the
// number of cells per block
void HOG::setCellDimensions()
{
	cellHeight = DET_WINDOW_H / CELL_NUMBER_Y;
	cellWidth = DET_WINDOW_W / CELL_NUMBER_X;

	if (remainder(DET_WINDOW_H, CELL_NUMBER_Y) != 0 || remainder(DET_WINDOW_W, CELL_NUMBER_X) != 0)
		cout << "WARNING: Remainder between cell dimensions is not 0. Sliding windows may cause problems" << endl;

	cout << "CELL WIDTH = " << cellWidth << " CELL HEIGHT = " << cellHeight << endl;
}

Mat HOG::getHOG(Mat image)
{
	//Resize image so it fits with the planned window dimension
	Mat imageCopy = image.clone();
	resize(imageCopy, imageCopy, Size(DET_WINDOW_W, DET_WINDOW_H));
	imshow("RESIZED IMAGE", imageCopy);
	waitKey(0);

	//Define a <vector<vector>> Mat containing all cells descriptors in an x,y manner 
	vector<vector<Mat>> cellHistograms(CELL_NUMBER_X, vector<Mat>(CELL_NUMBER_Y));

	//Define a rectangle which will contain the image cell
	Rect cellCrop;

	//Step 1: Get the HOG of every cell
	for (int i = 0; i < CELL_NUMBER_X; i++)
	{
		for (int j = 0; j < CELL_NUMBER_Y; j++)
		{
			cellCrop = Rect(i*cellWidth, j*cellHeight, cellWidth, cellHeight);
			cellHistograms[i][j] = getCellHistogram(imageCopy(cellCrop));
			//cout << "CELL HISTOGRAM OF " << i << " , " << j << " " << cellHistograms[i][j] << endl;
		}
	}

	//Step 2: Normalize the histograms in blocks defined in CELL_PER_BLOCK
	vector<vector<Mat>> normalizedCellHistograms = normalizeHistograms(cellHistograms);

	//Step X: Draw histograms
	drawHistograms(imageCopy, normalizedCellHistograms);

	return Mat();
}

Mat HOG::getCellHistogram(Mat cell) 
{
	//Show crop
	imshow("CELL", cell);
	waitKey(0);

	//Calculate gradients gx and gy
	// NOTE: I AM SUPPOSING TO HAVE A GRAYSCALE IMAGE
	Mat gx, gy;
	Sobel(cell, gx, CV_32F, 1, 0, 1);
	Sobel(cell, gy, CV_32F, 0, 1, 1);

	// Calculate gradient angle and magnitude
	Mat ang, mag;
	cartToPolar(gx, gy, mag, ang, true);

	//cout << ang << endl;
	//cout << ang.type() << endl;
	if(!HISTOGRAMS_360) 
	{
		for (int i = 0; i < ang.rows; i++)
		{
			for (int j = 0; j < ang.cols; j++)
			{
				if (ang.at<float>(i, j) > 180.0f)
				{
					ang.at<float>(i, j) -= 180.0f;
				}
			}
		}
	}
	

	// Histogram Storage
	Mat histogram = Mat::zeros(binValues.size(), CV_32F);
	float tempAng;
	float valueLocation = 0;
	float magnitudeToNextBin;
	float magnitudeToPreviousBin;

	for (int i = 0; i < cell.rows; i++)
	{
		for (int j = 0; j < cell.cols; j++)
		{
			for (int k = 0; k < histogram.rows; k++)
			{
				if (k != 0 && k != (histogram.rows - 1))
				{
					//Check to which bin does the gradient angle in the pixel i,j belong
					if (ang.at<float>(i, j) >= binValues.at<float>(0, k) && ang.at<float>(i, j) < binValues.at<float>(0, k + 1))
					{
						// Distribute the magnitude between the two bins:
						// Scale the angle at i,j between 0-20
						tempAng = ang.at<float>(i, j) - binValues.at<float>(0, k);
						// Get the % location in the value ranges
						// e.g. if angle = 45 and the bin limits are 40-60 then valueLocation = (45-40)/20
						// meaning that 5/20 of the magnitud value goes for bin 60 and (1-5/20) of the value goes for bin 40
						valueLocation = tempAng / binSize;
						magnitudeToNextBin = mag.at<float>(i, j)*(valueLocation);
						magnitudeToPreviousBin = mag.at<float>(i, j)*(1 - valueLocation);

						//Collect the values in the histogram
						histogram.at<float>(0, k+1) += magnitudeToNextBin;
						histogram.at<float>(0, k) += magnitudeToPreviousBin;
					}
				} else if (k == 0 && ang.at<float>(i, j) < binValues.at<float>(0, k+1))
				{
					tempAng = ang.at<float>(i, j);
					valueLocation = tempAng / binSize;
					magnitudeToNextBin = mag.at<float>(i, j)*(valueLocation);
					magnitudeToPreviousBin = mag.at<float>(i, j)*(1 - valueLocation);
					histogram.at<float>(0, k+1) += magnitudeToNextBin;
					histogram.at<float>(0, k) += magnitudeToPreviousBin;
				} else if (k == (histogram.rows - 1) && ang.at<float>(i, j) >= binValues.at<float>(0, k))
				{
					tempAng = ang.at<float>(i, j) - binValues.at<float>(0, k);
					valueLocation = tempAng / binSize;
					magnitudeToNextBin = mag.at<float>(i, j)*(valueLocation);
					magnitudeToPreviousBin = mag.at<float>(i, j)*(1 - valueLocation);
					histogram.at<float>(0, 0) += magnitudeToNextBin;
					histogram.at<float>(0, k) += magnitudeToPreviousBin;
				}

			}
		}
	}

	return histogram;
}

//NOTE: ASSUMING 2x2 BLOCKS
vector<vector<Mat>> HOG::normalizeHistograms(vector<vector<Mat>> cellHistograms)
{
	Mat blockHistogram;
	vector<vector<Mat>> normalizedCells(CELL_NUMBER_X, vector<Mat>(CELL_NUMBER_Y));

	for (int i = 0; i < CELL_NUMBER_X; i++)
	{
		for (int j = 0; j < CELL_NUMBER_Y; j++)
		{
			//cout << "ANALYZING CELL " << i << " " << j << endl;
			blockHistogram = cellHistograms[i][j].clone();

			//If we are on the X limit take the X-1 cell
			if (i + 1 != CELL_NUMBER_X) 
			{
				vconcat(blockHistogram, cellHistograms[i + 1][j].clone(), blockHistogram);
			}
			else
			{
				vconcat(blockHistogram, cellHistograms[i - 1][j], blockHistogram);
			}

			//If we are on the Y limit take the Y-1 cell
			if (j + 1 != CELL_NUMBER_Y)
			{
				vconcat(blockHistogram, cellHistograms[i][j + 1], blockHistogram);
			}
			else
			{
				vconcat(blockHistogram, cellHistograms[i][j - 1], blockHistogram);
			}

			//If we are on the top-right corner take the X-1,Y+1 cell
			if (j == 0 && i + 1 == CELL_NUMBER_X)
			{
				vconcat(blockHistogram, cellHistograms[i - 1][j + 1], blockHistogram);
			}

			//If we are on the bottom-left corner take the X+1, Y-1 cell
			if (j + 1 == CELL_NUMBER_Y  && i + 1 != CELL_NUMBER_X)
			{
				vconcat(blockHistogram, cellHistograms[i + 1][j - 1], blockHistogram);
			}

			//If we are on the bottom-right corner take the X-1,Y-1 cell
			if (j + 1 == CELL_NUMBER_Y && i + 1 == CELL_NUMBER_X) 
			{
				vconcat(blockHistogram, cellHistograms[i - 1][j - 1], blockHistogram);
			}

			//If we are not in a corner, take the X+1,Y+1 cell
			if (j + 1 != CELL_NUMBER_Y && i + 1 != CELL_NUMBER_X)
			{
				vconcat(blockHistogram, cellHistograms[i + 1][j + 1], blockHistogram);
			}
	
			normalize(blockHistogram,blockHistogram,1,NORM_L2);

			//Take the first cell
			blockHistogram.resize(BIN_NUMBER);
			//cout << blockHistogram << endl;
			normalizedCells[i][j] = blockHistogram;
		}
	}


	return normalizedCells;
}


Mat HOG::getBlockHistogram(Mat block)
{

	// List of histograms of the cells that compose the block
	vector<Mat> cellHistograms;
	//Region to be cropped in the block
	Rect cellCrop;
	for (int i = 0; i < CELL_PER_BLOCK_X; i++)
	{
		for (int j = 0; j < CELL_PER_BLOCK_Y; j++)
		{
			cellCrop = Rect(i*cellWidth, j*cellHeight, cellWidth, cellHeight);
			cellHistograms.push_back(getCellHistogram(block(cellCrop)));
		}
	}

	//Concatenate all histograms into a single descriptor
	Mat blockHistogram;
	for (int i = 0; i < cellHistograms.size(); i++) 
	{
		if (i == 0) blockHistogram = cellHistograms[i];
		else vconcat(blockHistogram, cellHistograms[i], blockHistogram);
	}

	normalize(blockHistogram,blockHistogram,NORM_L2);
	//cout << blockHistogram << endl;
	return blockHistogram;
}

void HOG::drawHistograms(Mat image, vector<vector<Mat>> histograms)
{
	Mat display = image.clone();
	cvtColor(display, display, COLOR_GRAY2BGR);
	namedWindow("HISTOGRAMS", WINDOW_NORMAL);
	Point p1, p2;
	float dX, dY;
	float P = 50.0f; //to enlage arrows 
	const float toRadians = PI / 180.0f;

	for (int i = 0; i < CELL_NUMBER_X; i++)
	{
		for (int j = 0; j < CELL_NUMBER_Y; j++)
		{
			p1 = Point(i*cellWidth + cellWidth/2, j*cellHeight + cellHeight/2);
			for (int k = 0; k < BIN_NUMBER; k++)
			{
				//cout << "ANGLE " << binValues.at<float>(k, 0) << " : ";
				//cout << sin(binValues.at<float>(k, 0)*toRadians) << " COS " << cos(binValues.at<float>(k, 0)*toRadians) << endl;

				dY = sin(binValues.at<float>(k, 0)*toRadians)*histograms[i][j].at<float>(k, 0);
				dX = cos(binValues.at<float>(k, 0)*toRadians)*histograms[i][j].at<float>(k, 0);

				//cout << "DX: " << dX*20 << " DY: " << dY*20 << endl;
				p2 = Point(i*cellWidth + cellWidth / 2 + dX*P, j*cellHeight + cellHeight / 2 + dY*P);
				arrowedLine(display, p1, p2, Scalar(255, 0, 0));
			}
		}
	}

	imshow("HISTOGRAMS", display);
	waitKey(0);
	destroyWindow("HISTOGRAMS");

}


// Training functions

void HOG::beginTraining(string filePath)
{
	vector<String> fileNames;
	vector<Mat> histogramFiles;

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
		vector<Mat> ROI = getROI(src, abort);
	}
}

// Allows the user to select rectangles from the image
// and calculate their histograms.
vector<Mat> HOG::getROI(Mat image, bool &abort)
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
	setMouseCallback(window,onMouse,&ROI);

	Mat imageCopy; 
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
			case 27 :
				esc = true;
				break;

		}

		//If ESC was pushed finish
		if (esc) break;
	}

	// Impede any more inputs 
	setMouseCallback(window, NULL, NULL);
	destroyWindow(window);

	// Histograms of the ROIs
	vector<Mat> histogramsROI;

	//Get the histograms of every ROI
	for (size_t i = 0; i < ROIs.size(); i++)
	{
		histogramsROI.push_back(getHOG(image(ROIs[i])));
	}

	cout << "DONEZO!" << endl;
	return histogramsROI;
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
		Rect* selection = (Rect*) userdata;
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
