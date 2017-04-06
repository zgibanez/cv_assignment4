#include "performance.h"

using namespace cv;
using namespace std;

void writeAUC()
{
	Tester ts = Tester(); 
	Trainer tr = Trainer();

	Mat AUC = Mat::zeros(Size(3, 1), CV_32F);

	for (double c = 0.1; c < 10000000000; c *= 10)
	{
		Mat PR = Mat::zeros(Size(3, 1), CV_32F);
		PR.at<float>(0, 2) = (float)c;
		cout << "For C =" << (float)c << endl;
		tr.svm.setParams(c, 0.1, 1, ml::SVM::LINEAR);
		tr.train(60,140);
		ts.setSVM(tr.svm);
		ts.test("dataset\\testing\\",true,PR);

		vconcat(AUC, PR.clone(), AUC);
	}

	cout << AUC << endl;

	FileStorage fs("AUC.xml", FileStorage::WRITE);
	fs << "AUC" << AUC;
	fs.release();
}

