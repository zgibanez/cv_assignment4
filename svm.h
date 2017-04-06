#include <opencv2/core/core.hpp>
#include <stddef.h>
#include <vector>
#include <opencv2/ml/ml.hpp>

class SVM
{
	cv::Ptr<cv::ml::SVM> machine;

	public:
		SVM();
		virtual ~SVM();

		void setParams(double c, double nu, int degree, int kernel = cv::ml::SVM::POLY);
		cv::Ptr<cv::ml::SVM> getSvm()
		{
			return machine;
		}

		void setTrainedMachine(cv::Ptr<cv::ml::SVM> tm)
		{
			machine = tm;
		}
		
};
