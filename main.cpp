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

	//trainer.buildROISet("dataset\\");
	//trainer.buildROISet("dataset\\negative_training\\");
	trainer.buildHOGSet("dataset\\roi\\","positive");
	trainer.buildHOGSet("dataset\\roi_n\\","negative");
	trainer.train(40);
	/*trainer.crossValidation(5,1,0.5,3);
	double c = 0.00001;
	double nu = 0.1;
	for (int degree = 1; degree < 6; degree++)
	{
		c = 0.00001;
		for (int i = 0; i < 20; i++)
		{
			nu = 0.1;
			for (int j = 0; j < 10; j++)
			{
				trainer.crossValidation(5, c, nu, degree);
				nu += 0.1;
			}
			c *= 10;
		}
	}*/



	Tester tester = Tester();
	tester.loadTSVM("trained_svm.xml");
	tester.setSVM(trainer.getTrainedSVM());
	//tester.test("dataset\\testing\\");

	Mat img = imread("1.jpg", IMREAD_GRAYSCALE);
	//cout << a.rows << endl;
	tester.getHeatMap(img,true);

	return;
}