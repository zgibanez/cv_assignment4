#include "tester.h"

#define SHEEP_LOCATIONS_FILE "sheep_locations.xml"
#define IOU_THRESHOLD 0.2f

void writeAUC(int positive_sample, int negative_sample);
void getObjectLocations(std::string fileDirectory);
void getFPPW(std::string fileDirectory, bool show = false);
void getAspectRatio();

