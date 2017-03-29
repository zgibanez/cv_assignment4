#include "tester.h"
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace std;
using namespace cv;

void main()
{
	//Calculate gradients gx and gy
	Trainer trainer = Trainer();

	//trainer.buildROISet("dataset\\positive_training\\");
	//trainer.buildROISet("dataset\\negative_training\\");
	//trainer.buildHOGSet("dataset\\roi\\","positive");
	//trainer.buildHOGSet("dataset\\roi_n\\","negative");
	trainer.train(10);
	
	Tester tester = Tester();
	//tester.loadTSVM("trained_svm.xml");
	tester.setSVM(trainer.getTrainedSVM());
	tester.test("dataset\\testing\\");

	return;
}