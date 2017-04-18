#include "hog.h"

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

	//cout << "Bin values are: " << binLimits << endl;
	binValues = binLimits;
	binSize = degreesPerBin;
}

// Set dimensions in pixels of the cells given the
// number of cells per block
void HOG::setCellDimensions()
{
	cellHeight = DET_WINDOW_H / CELL_NUMBER_Y;
	cellWidth = DET_WINDOW_W / CELL_NUMBER_X;

	//if (remainder(DET_WINDOW_H, CELL_NUMBER_Y) != 0 || remainder(DET_WINDOW_W, CELL_NUMBER_X) != 0)
		//cout << "WARNING: Remainder between cell dimensions is not 0. Sliding windows may cause problems" << endl;

	//cout << "CELL WIDTH = " << cellWidth << " CELL HEIGHT = " << cellHeight << endl;
}

Mat HOG::getHOG(Mat image, bool debug)
{
	//Resize image so it fits with the planned window dimension
	Mat imageCopy = image.clone();
	resize(imageCopy, imageCopy, Size(DET_WINDOW_W, DET_WINDOW_H));
	GaussianBlur(imageCopy, imageCopy, Size(5, 5), 0);

	if (debug)
	{
		imshow("RESIZED IMAGE", imageCopy);
		waitKey(0);
	}

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
	if (debug)  drawHistograms(imageCopy, normalizedCellHistograms);

	return concatNormalizedCellHistograms(normalizedCellHistograms);
}

Mat HOG::getCellHistogram(Mat cell) 
{
	//Show crop
	//imshow("CELL", cell);
	//waitKey(0);

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

			//If we are on the bottom take the X+1, Y-1 cell
			if (j + 1 == CELL_NUMBER_Y  && i + 1 != CELL_NUMBER_X)
			{
				vconcat(blockHistogram, cellHistograms[i + 1][j - 1], blockHistogram);
			}

			//If we are on the right take the X-1, Y-1 cell
			if (j + 1 != CELL_NUMBER_Y && i + 1 == CELL_NUMBER_X)
			{
				vconcat(blockHistogram, cellHistograms[i - 1][j + 1], blockHistogram);
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

Mat HOG::concatNormalizedCellHistograms(vector<vector<Mat>> nch)
{
	Mat histList = nch[0][0].clone();

	for (int m = 0; m < CELL_NUMBER_Y; m++)
	{
		for (int n = 1; n < CELL_NUMBER_X; n++)
		{
			vconcat(histList, nch[m][n], histList);
		}
	}

	return histList;
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