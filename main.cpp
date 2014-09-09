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

    int nValues = 20000;
    double *values = new double[nValues];
    for(int i = 0; i < nValues; i++)
    {
	values[i] = (double) gsl_ran_gaussian(r, 1.0);
	//values[i] = ((double)rand())/(double)RAND_MAX;
    }

    myPDF *newPDF = new myPDF(50, values, nValues);
    for(int i = 0 ; i < 2000; i++ )
    {
	cout << newPDF->drawRandom( ((double)rand())/(double)RAND_MAX )  << "\n";
    }

    delete newPDF;
    gsl_rng_free(r);
    return 0;
}
