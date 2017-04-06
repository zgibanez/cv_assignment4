#include "trainer.h"

#define WINDOW_STEP 0.2
#define MAXIMUM_OVERLAP 0.2
#define WINDOW_WIDTH 110
#define WINDOW_HEIGHT 80
#define POSITIVE_BIAS 0.4f

typedef struct {
	float scale;
	float x;
	float y;
	float score;
	bool flipped;
} Match;

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
		void Tester::test(std::string imgDir, bool returnPR = false, cv::Mat &PR = cv::Mat());
		std::vector<Match> Tester::getPositiveMatches(cv::Mat image, bool show = false);
		std::vector<Match> Tester::adjustBoundingBoxes(std::vector<Match> results,cv::Mat image);
		float Tester::getCollision(Match a, Match b, cv::Mat image = cv::Mat());
		std::vector<Match> Tester::applyNMS(std::vector<Match> results, cv::Mat image = cv::Mat());
		void Tester::drawPositiveMatchBB(std::vector<Match> results, cv::Mat image);
		void Tester::detect(cv::Mat image, bool show = false);


};



