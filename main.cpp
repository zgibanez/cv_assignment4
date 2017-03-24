#include "HOGClass.h"
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace std;
using namespace cv;

void main()
{
	/*string imagePath = "org/";
	for(int i = 1 ; i <= 200 ; i++)
	{
		
		Mat image = imread(imagePath + to_string(i) + ".pgm");
		resize(image, image, Size(image.cols * 3, image.rows * 3));
		string test = imagePath + to_string(i) + ".pgm";
		cout << test << endl;
		imshow("IMAGE", image);
		waitKey(0);
	}*/
	
	//Calculate gradients gx and gy
	HOG* hog = new HOG();


	Mat cell = imread("C:/Users/pey_l/OneDrive/Imágenes/roland_geraerts_3.jpg",CV_LOAD_IMAGE_GRAYSCALE);
	//hog.getBlockHistogram(cell);
	hog->getHOG(cell);
	
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