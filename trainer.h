#include "hog.h"
#include "svm.h"

#define SAMPLE_NUMBER 50

//GUI functions
void onMouse(int event, int x, int y, int flags, void *userdata);
void help();

class Trainer
{
	HOG hog;
	SVM svm;

	public:
		Trainer();
		virtual ~Trainer();
		SVM Trainer::getTrainedSVM();

		//Training
		void Trainer::buildROISet(std::string filePath);
		std::vector<cv::Mat> Trainer::getROI(cv::Mat image, bool &abort, std::string fileName);
		void Trainer::buildHOGSet(std::string imgDir, std::string setName);
		cv::Mat Trainer::takeHOGSampleFromFile(std::string filename,int offset,int size);
		void Trainer::train(int sample_size);
		//return the trained SVM
		float Trainer::crossValidation(int fold_number, double c, double nu, int degree);
};
