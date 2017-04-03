#include "svm.h"

using namespace cv;
using namespace std;

SVM::SVM()
{
	machine = ml::SVM::create();
}

void SVM::setParams(double c, double nu, int degree)
{
	machine->setType(ml::SVM::C_SVC);
	machine->setKernel(ml::SVM::LINEAR);
	machine->setGamma(3);
	//machine->setDegree(degree);
	machine->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, 10000, 1e-6));

	machine->setC(c);
	machine->setNu(nu);
}

SVM::~SVM()
{

}