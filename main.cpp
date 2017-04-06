#include "performance.h"
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace std;
using namespace cv;

void main()
{
	//TODO
	//2) Probar threshold negativo para asegurar que se tengan ovejas
	//3) Petar al trainer con negativas
	//4) Investigar las dimensiones de una oveja
	//5) En seleccion de ROI, investigar el porcentaje de imagen ocupado por la oveja(máximo, mínimo y promedio)
	//7) Hacer funcion para area of intersection (IoU)

	//writeAUC();

	Trainer trainer = Trainer();

	//trainer.buildROISet("dataset\\positive_training\\");
	//trainer.buildROISet("dataset\\negative_training\\");
	trainer.buildHOGSet("dataset\\roi\\","positive");
	trainer.buildHOGSet("dataset\\roi_n\\","negative");
	
	trainer.setOptimalParameters();
	//trainer.svm.setParams(1000, 0.1, 1, ml::SVM::LINEAR);
	trainer.train(60,140,true);
	
	Tester tester = Tester();
	tester.loadTSVM("trained_svm.xml");
	//tester.setSVM(trainer.getTrainedSVM());
	//tester.test("dataset\\testing\\");
	
	Mat img = imread("30.pgm", IMREAD_GRAYSCALE);
	tester.detect(img);
	//cout << a.rows << endl;
	
	
	return;
}