#include "HOGClass.h"

#include <opencv2/core/mat.hpp>
#include <opencv2/core/operations.hpp>
#include <opencv2/core/types_c.h>
#include <opencv2/imgproc.hpp>
#include <cassert>
#include <iostream>
#include <fstream>
#include <opencv2/highgui/highgui.hpp>



using namespace std;
using namespace cv;

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
	if (SIGN_HISTOGRAMS) totalDegrees = 360.0; else totalDegrees = 180.0;
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

Mat HOG::getHOG(Point pixel_center, Mat image)
{
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

Mat HOG::getBlockHistogram(Mat block)
{
	imshow("BLOCK", block);
	waitKey(0);
	// List of histograms of the cells that compose the block
	vector<Mat> cellHistograms;
	//Region to be cropped in the block
	Rect cellCrop;
	for (int i = 0; i < CELL_NUMBER_X; i++)
	{
		for (int j = 0; j < CELL_NUMBER_Y; j++)
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

	cout << blockHistogram << endl;
	return blockHistogram;
}

