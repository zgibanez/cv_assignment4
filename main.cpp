#include "tester.h"
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace std;
using namespace cv;

void main()
{
	//TODO
	//1) El primer tamaño probado debe ser el de la ventana
	//2) Probar threshold negativo para asegurar que se tengan ovejas
	//3) Petar al trainer con negativas
	//4) Investigar las dimensiones de una oveja
	//5) En seleccion de ROI, investigar el porcentaje de imagen ocupado por la oveja(máximo, mínimo y promedio)
	//	6) Preparar Area under the curve
	//	7) Hacer funcion para area of intersection (IoU)

	Trainer trainer = Trainer();

	//trainer.buildROISet("dataset\\positive_training\\");
	//trainer.buildROISet("dataset\\negative_training\\");
	//trainer.buildHOGSet("dataset\\roi\\","positive");
	//trainer.buildHOGSet("dataset\\roi_n\\","negative");
	trainer.setOptimalParameters();
	trainer.train(50);
	
	Tester tester = Tester();
	//tester.loadTSVM("trained_svm.xml");
	tester.setSVM(trainer.getTrainedSVM());
	//tester.test("dataset\\testing\\");
	
	Mat img = imread("2.jpg", IMREAD_GRAYSCALE);
	//tester.test("dataset\\testing\\");
	//cout << a.rows << endl;
	vector<Match> results = tester.getPositiveMatches(img,1);
	//tester.drawPositiveMatchBB(results, img.clone());
	vector<Match> matches = tester.applyNMS(results);

	tester.drawPositiveMatchBB(matches, img);
	
	return;
}