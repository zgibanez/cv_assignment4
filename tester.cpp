#include "tester.h"
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

Tester::Tester()
{
	hog = HOG();
	svm = SVM();
}

Tester::~Tester() {}

void Tester::loadTSVM(string svmFile)
{
	svm.getSvm()->load(svmFile);
}

void Tester::test(string imgDir)
{
	vector<String> fileNames;

	//Get the list of files
	glob(imgDir, fileNames);
	Mat response;

	Mat predicted_label(Size(1,1),CV_32F);

	for (size_t i = 0; i < fileNames.size(); i++)
	{
		Mat img = imread(fileNames[i], IMREAD_GRAYSCALE);
		Mat img_hog = hog.getHOG(img);
		transpose(img_hog, img_hog);
		cout << img_hog.type() << " cols: " << img_hog.cols << " rows: " << img_hog.rows << endl;
		svm.getSvm()->predict(img_hog,response,ml::SVM::RAW_OUTPUT);
		string answer = response.at<float>(0, 0) < 0.0f ? "yes" : "no";
		cout << fileNames[i]<<" es una oveja?: " << answer << " Distance " << abs(response.at<float>(0, 0)) << endl;
		imshow("Imagen", img);
		waitKey(0);
		destroyAllWindows();
	}
	//svm.getSvm()->predict();


}

Mat Tester::getHeatMap(Mat image, bool show)
{
	Rect r = Rect(0, 0, WINDOW_WIDTH, WINDOW_HEIGHT);
	Mat imageCopy;
	Mat imageTest;
	Mat tempHist;
	Mat out;
	Mat rawOut;
	float dist;
	vector<vector<float>> results;

	//For every scale
	for (float scale = 0.2f; scale < 1.2f; scale += 0.2f)
	{
		resize(image, imageTest, Size(image.cols*scale, image.rows*scale));
		imshow("IM",imageTest);
		waitKey(0);
		for (float i = 0.0f; i <= imageTest.rows - (WINDOW_HEIGHT); i += OVERLAP_WINDOW*WINDOW_HEIGHT)
		{
			for (float j = 0.0f; j <= (imageTest.cols - (WINDOW_WIDTH)); j += OVERLAP_WINDOW*WINDOW_WIDTH)
			{
				r.x = j;
				r.y = i;

				tempHist = hog.getHOG(imageTest(r));
				//TODO: PENSAR EN DONDE PONER EL TRANSPOSE PARA NO TENER QUE HACERLO REPETIDAMENTE
				transpose(tempHist, tempHist);
				//svm.getSvm()->predict(tempHist,out);
				svm.getSvm()->predict(tempHist, rawOut, ml::SVM::RAW_OUTPUT);
				dist = rawOut.at<float>(0, 0);
				if (dist < 0)
				{
					vector<float> tempResult(4);
					//Group results
					tempResult[0] = scale;
					tempResult[1] = r.x;
					tempResult[2] = r.y;
					tempResult[3] = abs(dist);
					cout << "Scale: " << scale << " X=" << r.x << " Y=" << r.y << " Score=" << abs(dist) << endl;
					//Store them 
					results.push_back(tempResult);
				}

				if (show)
				{
					imageCopy = imageTest.clone();
					rectangle(imageCopy, r, Scalar(0, 255, 0), 2);
					imshow("IMAGE", imageCopy);
					waitKey(0);
				}
			}
		}
	}


	return Mat();
}