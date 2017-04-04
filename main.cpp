#include "tester.h"
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace std;
using namespace cv;

void main()
{
	Trainer trainer = Trainer();

	//trainer.buildROISet("dataset\\positive_training\\");
	//trainer.buildROISet("dataset\\negative_training\\");
	//trainer.buildHOGSet("dataset\\roi\\","positive");
	//trainer.buildHOGSet("dataset\\roi_n\\","negative");
	trainer.setOptimalParameters();
	trainer.train(50);
	
	Tester tester = Tester();
	//tester.loadTSVM("trained_svm.xml");
	tester.setSVM(trainer.getTrainedSVM());
	//tester.test("dataset\\testing\\");
	
	Mat img = imread("4.jpg", IMREAD_GRAYSCALE);
	//cout << a.rows << endl;
	vector<Match> results = tester.getPositiveMatches(img,1);
	//tester.drawPositiveMatchBB(results, img.clone());
	vector<Match> matches = tester.applyNMS(results);

	tester.drawPositiveMatchBB(matches, img);
	
	return;
}