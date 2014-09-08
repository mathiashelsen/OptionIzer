#include <iostream>
#include <cstdlib>

#include "myPDF.hpp"

using namespace std;

int main(int argc, char **argv)
{
    int nValues = 10000;
    double *values = new double[nValues];
    for(int i = 0; i < nValues; i++)
    {
	values[i] = ((double)rand())/(double)RAND_MAX;
    }

    myPDF *newPDF = new myPDF(50, values, nValues);
    return 0;
}
