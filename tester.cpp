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
	float response;

	Mat predicted_label(Size(1,1),CV_32F);

	for (size_t i = 0; i < fileNames.size(); i++)
	{
		Mat img = imread(fileNames[i], IMREAD_GRAYSCALE);
		Mat img_hog = hog.getHOG(img);
		transpose(img_hog, img_hog);
		cout << img_hog.type() << " cols: " << img_hog.cols << " rows: " << img_hog.rows << endl;
		response = svm.getSvm()->predict(img_hog);
		cout << fileNames[i]<<" es una oveja?: " <<response << endl;
		imshow("Imagen", img);
		waitKey(0);
		destroyAllWindows();
	}
	//svm.getSvm()->predict();
}