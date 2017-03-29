#include "trainer.h"

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

};
