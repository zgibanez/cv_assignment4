#include <opencv2/core/core.hpp>
#include <stddef.h>
#include <vector>
#include "hog.h"

#define HOG_FILE "positive_hog.xml"

//GUI functions
void onMouse(int event, int x, int y, int flags, void *userdata);
void help();

class Trainer
{
	HOG hog;

	public:
		Trainer();
		virtual ~Trainer();
		//Training
		void Trainer::buildROISet(std::string filePath);
		std::vector<cv::Mat> Trainer::getROI(cv::Mat image, bool &abort, std::string fileName);
		void Trainer::buildHOGSet(std::string imgDir);
};
