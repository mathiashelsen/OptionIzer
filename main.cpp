#include <iostream>
#include <cstdlib>

#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>

#include "myPDF.hpp"
#include "timeSeries.hpp"

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
    timeSeries *newSeries = new timeSeries(2000, 2000, newPDF);

    cout << newSeries->series[0][0] << ", " << newSeries->series[0][1] << ", " << newSeries->series[1][0] << "\n";

    delete newSeries;
    delete newPDF;
    gsl_rng_free(r);
    return 0;
}
