#include "hog.h"
#include "svm.h"

#define SAMPLE_NUMBER 60

//GUI functions
void onMouse(int event, int x, int y, int flags, void *userdata);
void help();

class Trainer
{
	public:
		HOG hog;
		SVM svm;

		Trainer();
		virtual ~Trainer();
		SVM Trainer::getTrainedSVM();

		//Training
		void Trainer::buildROISet(std::string filePath);
		std::vector<cv::Mat> Trainer::getROI(cv::Mat image, bool &abort, std::string fileName);
		void Trainer::buildHOGSet(std::string imgDir, std::string setName);
		cv::Mat Trainer::takeHOGSampleFromFile(std::string filename,int offset,int size);
		void Trainer::train(int pos_sample_size, int neg_sample_size, bool saveSVM = false);
		//return the trained SVM
		float Trainer::crossValidation(int fold_number, double c, double nu, int degree);
		void Trainer::setOptimalParameters();
};
