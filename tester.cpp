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

vector<Match> Tester::getPositiveMatches(Mat image, bool show)
{
	Rect r = Rect(0, 0, WINDOW_WIDTH, WINDOW_HEIGHT);
	Mat imageCopy;
	Mat imageTest;
	Mat tempHist;
	Mat out;
	Mat rawOut;
	float dist;
	bool flipped = false;
	vector<Match> results;

	//For every scale
	for (int flipC = 0; flipC <= 1; flipC++)
	{
		flipped = !flipped;

		for (float scale = 0.2f; scale < 1.2f; scale += 0.2f)
		{

			resize(image, imageTest, Size(image.cols*scale, image.rows*scale));
			if (flipped)
			{
				flip(imageTest, imageTest,1);
			}
			if (show)
			{
				imshow("IM", imageTest);
				waitKey(0);
			}
			for (float i = 0.0f; i <= imageTest.rows - (WINDOW_HEIGHT); i += OVERLAP_WINDOW*WINDOW_HEIGHT)
			{
				for (float j = 0.0f; j <= (imageTest.cols - (WINDOW_WIDTH)); j += OVERLAP_WINDOW*WINDOW_WIDTH)
				{
					r.x = j;
					r.y = i;

					tempHist = hog.getHOG(imageTest(r));
					//TODO: PENSAR EN DONDE PONER EL TRANSPOSE PARA NO TENER QUE HACERLO REPETIDAMENTE
					transpose(tempHist, tempHist);
					svm.getSvm()->predict(tempHist, rawOut, ml::SVM::RAW_OUTPUT);
					dist = rawOut.at<float>(0, 0);
					if (dist < 0)
					{
						//vector<float> tempResult(5);
						Match tempResult;
						//Group results
						tempResult.scale = scale;
						tempResult.flipped = flipped;
						tempResult.x = r.x;
						tempResult.y = r.y;
						tempResult.score = abs(dist);
						//cout << "Scale: " << scale << " X=" << r.x << " Y=" << r.y << " Score=" << abs(dist) << endl;
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
	}
	
	vector<Match> final_results = adjustBoundingBoxes(results,image);
	//drawPositiveMatchBB(final_results, image);
	return final_results;
}

vector<Match> Tester::adjustBoundingBoxes(vector<Match> results,Mat image)
{
	vector<Match> adjusted;
	float x, y;
	float scale;
	float width, height;
	bool flipped;
	for (int i = 0; i < results.size(); i++)
	{
		Match current;
		flipped = results[i].flipped;
		scale = results[i].scale;
		width = WINDOW_WIDTH / scale;
		height = WINDOW_HEIGHT / scale;
		x = flipped ? (image.cols - results[i].x / scale - width) : results[i].x / scale;
		y = results[i].y / scale;
		current.flipped = false;
		current.scale = scale;
		current.x = x;
		current.y = y;
		current.score = results[i].score;
		adjusted.push_back(current);
	}
	return adjusted;
}

bool compareMatches(Match i, Match j)
{
	return i.score > j.score;
}

float Tester::getCollision(Match a, Match b, Mat image)
{
	//Check if there is collision
	if (a.x < b.x + WINDOW_WIDTH / b.scale &&
		a.x + WINDOW_WIDTH / a.scale > b.x &&
		a.y < b.y + WINDOW_HEIGHT / b.scale &&
		WINDOW_HEIGHT / a.scale + a.y > b.y) 
	{

	//Dimentions intersection area (rectangle)
		float w, h;
		if (a.x <= b.x)
		{
			w = a.x + WINDOW_WIDTH / a.scale - b.x;
		}
		else
		{
			w = b.x + WINDOW_WIDTH / b.scale - a.x;
		}

		if (a.y <= b.y)
		{
			h = a.y + WINDOW_HEIGHT / a.scale - b.y;
		}
		else
		{
			h = b.y + WINDOW_HEIGHT / b.scale - a.y;
		}

		//Get the percentage of area of a that h*b covers
		float areaIntersection = h*w;
		float areaNMS = WINDOW_WIDTH / b.scale * WINDOW_HEIGHT / b.scale;

		if (!image.empty())
		{
			Mat imageCopy = image.clone();
			cvtColor(imageCopy, imageCopy, COLOR_GRAY2BGR);
			Rect r1 = Rect(a.x, a.y, WINDOW_WIDTH / a.scale, WINDOW_HEIGHT / a.scale);
			Rect r2 = Rect(b.x, b.y, WINDOW_WIDTH / b.scale, WINDOW_HEIGHT / b.scale);
			rectangle(imageCopy, r1, Scalar(0, 0, 255), 2);
			rectangle(imageCopy, r2, Scalar(255, 0, 255), 2);
			imshow("INTERSECTION", imageCopy);
			waitKey(0);
		}

		return areaIntersection/areaNMS;
	}
	else
	{
		return 0;
	}
}

vector<Match> Tester::applyNMS(vector<Match> results, Mat image)
{
	vector<Match> nms;
	std::sort(results.begin(),results.end(),compareMatches);
	for (int i = 0; i < results.size(); i++)
	{
		cout << " Score " << results[i].score << endl;
	}

	nms.push_back(results[0]);

	for (int i = 1; i < results.size(); i++)
	{
		bool t = false;
		for (int j = 0; j < nms.size(); j++)
		{
			if (getCollision(results[i], nms[j], image) > 0.2)
			{
				//cout << "COLISION SUPERO 40% " << endl;
				t = true;
				break;
			}

		}
		if (!t)
			nms.push_back(results[i]);
	}

	return nms;
}

void Tester::drawPositiveMatchBB(vector<Match> results, Mat image)
{
	Mat display;
	cvtColor(image,display, COLOR_GRAY2BGR);
	Rect r;
	float x, y;
	float scale;
	float width, height;
	bool flipped;
	for (int i = 0; i < results.size(); i++)
	{
		/*flipped = results[i].flipped;*/
		scale = results[i].scale;
		width = WINDOW_WIDTH / scale;
		height = WINDOW_HEIGHT / scale;
		/*x = flipped ? (image.cols - results[i].x/scale - width) : results[i].x / scale;
		y = results[i].y / scale;*/
		r = Rect(results[i].x, results[i].y, width, height);
		rectangle(display, r, Scalar(0, 230*results[i].score, 0), 2);
	}

	imshow("MATCHES", display);
	waitKey(0);
	//TODO: DRAW SCORE
}