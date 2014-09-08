#include "myPDF.hpp"

myPDF::myPDF(int nBins, double *x, int nValues)
{
    // Local variables to be freed after initializing
    double * tmpHisto = new double[nBins];
    double maxValue = x[0], minValue = x[0];
    // Allocating some memory for the inverse CDF
    binValues = new double[nBins];

    for(int i = 0; i < nValues; i++)
    {
	maxValue = (x[i] > maxValue) ? x[i] : maxValue;
	minValue = (x[i] < minValue) ? x[i] : minValue;
    }

    binSpacing = (maxValue - minValue)/((double) nBins);

    for(int i = 0; i < nValues; i++ )
    {
	tmpHisto[ (int) (x[i]/binSpacing) ]++;
    }

    for(int i = 0; i < nBins; i++)
    {
	std::cout << binSpacing * (double)i << "\t" << tmpHisto[i] << "\n";
    }

    delete[] tmpHisto;
}

myPDF::~myPDF()
{
    delete[] binValues;
}
