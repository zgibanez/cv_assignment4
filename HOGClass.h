#include <opencv2/core/core.hpp>
#include <stddef.h>
#include <vector>

// Window dimensions
#define DET_WINDOW_W 500
#define DET_WINDOW_H 200

//Are the histogram values signed or not?
#define SIGN_HISTOGRAMS true 

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

class HOG
{
	//Members
    cv::Mat binValues;
	float binSize;
	float cellHeight, cellWidth;

	//Methods
	public:
		virtual ~HOG();
		void HOG::setBinValues();
		void HOG::setCellDimensions();
		cv::Mat HOG::getCellHistogram(cv::Mat cell);
		cv::Mat HOG::getBlockHistogram(cv::Mat block);
		cv::Mat HOG::getHOG(cv::Mat image);
		std::vector<std::vector<cv::Mat>> HOG::normalizeHistograms(std::vector<std::vector<cv::Mat>> cellHistograms);

};

