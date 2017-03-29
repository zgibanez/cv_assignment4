#include "svm.h"

using namespace cv;
using namespace std;

SVM::SVM()
{
	machine = ml::SVM::create();
}

void SVM::setParams(int type, int kernel,int gamma)
{
	machine->setType(type);
	machine->setKernel(kernel);
	machine->setGamma(gamma);
	machine->setDegree(3);
	machine->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, 100, 1e-6));
}

SVM::~SVM()
{

}