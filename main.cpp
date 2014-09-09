#include <iostream>
#include <cstdlib>

#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>

#include "myPDF.hpp"

using namespace std;

int main(int argc, char **argv)
{
    const gsl_rng_type * T;
    gsl_rng * r;

    gsl_rng_env_setup();

    T = gsl_rng_default;
    r = gsl_rng_alloc (T);

    int nValues = 20000000;
    double *values = new double[nValues];
    for(int i = 0; i < nValues; i++)
    {
	values[i] = (double) gsl_ran_gaussian(r, 1.0);
	//values[i] = ((double)rand())/(double)RAND_MAX;
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
    
    double value = 0.0;
    for(int i = -500 ; i < 500 ; i++ )
    {
	value = newPDF->getCDFValue(0.02 * (double)i);
	cout << 0.02*(double)i << "\t" << value << "\n";
    } 

    delete newPDF;
    gsl_rng_free(r);
    return 0;
}
