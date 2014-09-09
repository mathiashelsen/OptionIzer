#include "myPDF.hpp"

myPDF::myPDF(int nBins, double *x, int nValues)
{
    // Local variables to be freed after initializing
    double * tmpHisto = new double[nBins];
    double * tmpCDF = new double[nBins];
    double maxValue = x[0], minValue = x[0];
    // Allocating some memory for the inverse CDF
    binValues = new double[nBins];
    // Find minimal and maximal values
    for(int i = 0; i < nValues; i++)
    {
	maxValue = (x[i] > maxValue) ? x[i] : maxValue;
	minValue = (x[i] < minValue) ? x[i] : minValue;
    }
    // from these deduce the spacing between bins
    binSpacing = (maxValue - minValue)/((double) nBins);
    // so you can create the histogram
    for(int i = 0; i < nValues; i++ )
    {
	tmpHisto[ (int) (x[i]/binSpacing) ]++;
    }
    // from this the cumulative PDF
    tmpCDF[0] = tmpHisto[0] / (double)nValues;
    for(int i = 1; i < nBins; i++)
    {
	tmpCDF[i] = tmpCDF[i-1]/(double)nValues + tmpHisto[i];
    }

    for(int i = 0; i < nBins; i++)
    {
	std::cout << binSpacing * (double)i << "\t" << tmpHisto[i] << "\n";
    }
   
    delete[] tmpCDF; 
    delete[] tmpHisto;
}

myPDF::~myPDF()
{
    delete[] binValues;
}
