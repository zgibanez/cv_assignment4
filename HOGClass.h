#include <opencv2/core/core.hpp>
#include <stddef.h>
#include <vector>

// Window dimensions
#define DET_WINDOW_W 500
#define DET_WINDOW_H 300

//Are the histogram values signed or not?
#define HISTOGRAMS_360 false 

// Number of bins in the histogram
#define BIN_NUMBER 9

// Number of cells in which the image is divided
#define CELL_NUMBER_X 4
#define CELL_NUMBER_Y 4

//Number of cells per block
#define CELL_PER_BLOCK_X 2
#define CELL_PER_BLOCK_Y 2

//Overlap percentage between blocks
#define OVERLAP_PERCENTAGE 0.5

//Other constants
#define PI 3.14159265

//GUI functions
void onMouse(int event, int x, int y, int flags, void *userdata);
void help();

class HOG
{
	//Members
    cv::Mat binValues;
	float binSize;
	float cellHeight, cellWidth;

	//Methods
	public:
		HOG();
		virtual ~HOG();
		void HOG::setBinValues();
		void HOG::setCellDimensions();
		cv::Mat HOG::getCellHistogram(cv::Mat cell);
		cv::Mat HOG::getBlockHistogram(cv::Mat block);
		cv::Mat HOG::getHOG(cv::Mat image);
		std::vector<std::vector<cv::Mat>> HOG::normalizeHistograms(std::vector<std::vector<cv::Mat>> cellHistograms);

		//Drawing
		void HOG::drawHistograms(cv::Mat image, std::vector<std::vector<cv::Mat>> histograms);

		//Training
		void HOG::beginTraining(std::string filePath);
		std::vector<cv::Mat> HOG::getROI(cv::Mat image, bool &abort);
};