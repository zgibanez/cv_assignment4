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

		void setParams(int type, int kernel, int gamma);
		cv::Ptr<cv::ml::SVM> getSvm()
		{
			return machine;
		}

		void setTrainedMachine(cv::Ptr<cv::ml::SVM> tm)
		{
			machine = tm;
		}
		
};
