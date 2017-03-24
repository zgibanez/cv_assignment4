#include "HOGClass.h"
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace std;
using namespace cv;

void main()
{
	//Calculate gradients gx and gy
	HOG hog;
	hog.setBinValues();
	hog.setCellDimensions();


	Mat cell = imread("D:/images/c1.jpg",CV_LOAD_IMAGE_GRAYSCALE);
	//hog.getBlockHistogram(cell);
	hog.getHOG(cell);
	
	//cell.convertTo(cell, CV_32F, 1 / 255.0);
	Mat gx, gy;
	Sobel(cell, gx, CV_32F, 1, 0, 1);
	Sobel(cell, gy, CV_32F, 0, 1, 1);
	Mat mag, ang;
	cartToPolar(gx, gy, mag, ang, true);

	Mat test, test2;
	gy.convertTo(test, CV_8U);
	gy.convertTo(test2, CV_8U);
	test = test + test2;
	imshow("TEST",test);
	waitKey();

}