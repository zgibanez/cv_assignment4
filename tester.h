#include "trainer.h"

#define OVERLAP_WINDOW 0.25

#define WINDOW_WIDTH 130
#define WINDOW_HEIGHT 100

class Tester
{
	HOG hog;
	SVM svm;

	public:
		Tester();
		virtual ~Tester();

		void Tester::setSVM(SVM s)
		{
			svm = s;
		}

		void Tester::loadTSVM(std::string svmFile);
		void Tester::test(std::string imgDir);
		cv::Mat Tester::getHeatMap(cv::Mat image, bool show = false);


};
