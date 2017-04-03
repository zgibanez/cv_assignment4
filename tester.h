#include "trainer.h"

#define OVERLAP_WINDOW 0.1

#define WINDOW_WIDTH 130
#define WINDOW_HEIGHT 100

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
		void Tester::test(std::string imgDir);
		std::vector<Match> Tester::getPositiveMatches(cv::Mat image, bool show = false);
		std::vector<Match> Tester::adjustBoundingBoxes(std::vector<Match> results,cv::Mat image);
		float Tester::getCollision(Match a, Match b, cv::Mat image = cv::Mat());
		std::vector<Match> Tester::applyNMS(std::vector<Match> results, cv::Mat image = cv::Mat());
		void Tester::drawPositiveMatchBB(std::vector<Match> results, cv::Mat image);


};



