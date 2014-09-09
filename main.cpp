#include <iostream>
#include <cstdlib>

#include "myPDF.hpp"

using namespace std;

int main(int argc, char **argv)
{
    int nValues = 100000;
    double *values = new double[nValues];
    for(int i = 0; i < nValues; i++)
    {
	values[i] = ((double)rand())/(double)RAND_MAX;
    }

    myPDF *newPDF = new myPDF(50, values, nValues);
    for(int i = 0; i < 50; i++ )
    {
	cout << 0.02*(double)i << "\t" << newPDF->getCDFValue(i) << "\n";
    }

    cout << "\n\n";

    double minValue = 0.0, maxValue = 0.0, binSpacing = 0.0;
    newPDF->getExtents(&minValue, &maxValue, &binSpacing);
    cout << "#" << minValue << "\t" << maxValue << "\t" << binSpacing << "\n";
    
  
    for(int i = -100 ; i < 600 ; i++ )
    {
	cout << 0.002*(double)i << "\t" << newPDF->getCDFValue( 0.002 * (double)i) << "\n";
    } 

    delete newPDF;
    return 0;
}
