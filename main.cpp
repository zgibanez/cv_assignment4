#include "performance.h"
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace std;
using namespace cv;

void main()
{

	//STEP 1: Build positive and negative data.
	//trainer.buildROISet("dataset\\positive_training\\");
	//trainer.buildROISet("dataset\\positive_training\\", "dataset\\roi_n\\");
		
		//Locate sheep in testing images
		//getObjectLocations("nms_test//");

	//STEP 2: Extract descriptors from positive and negative data
	//trainer.buildHOGSet("dataset\\roi\\","positive");
	//trainer.buildHOGSet("dataset\\roi_n\\","negative");

	//STEP 3: Get best parameters possible with 10-fold cross validation
	Trainer trainer = Trainer();
	trainer.setOptimalParameters(true);

	//STEP 4 A: Train the SVM with best parameters obtained
	trainer.train(60, 300, true);
	Tester tester = Tester();
	tester.setSVM(trainer.getTrainedSVM());

	//STEP 4 B: Alternatively, load a previously stored SVM
	//tester.loadTSVM("trained_svm.xml");

	//STEP 5: Test single images
	//tester.test("dataset\\testing\\");

	//STEP 6 A: Test object detection (set of images)

		//False positives per window
		getFPPW("nms_test//");

		//Precision-Recall curve (with number of samples)
		writeAUC(40, 200);
	
	//STEP 6 B: Test object detection (single image)
	Mat img = imread("nms_test//5.pgm", IMREAD_GRAYSCALE);
	tester.detect(img,1);

	return;
}