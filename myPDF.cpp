#include "myPDF.hpp"

myPDF::myPDF(int _nBins, double *x, int nValues)
{
    nBins = _nBins;
    // Local variables to be freed after initializing
    double * tmpHisto = new double[nBins];
    maxValue = x[0], minValue = x[0];
    // Allocating some memory for the inverse CDF
    CDF = new double[nBins+1];
    // Find minimal and maximal values
    for(int i = 0; i < nValues; i++)
    {
	maxValue = (x[i] > maxValue) ? x[i] : maxValue;
	minValue = (x[i] < minValue) ? x[i] : minValue;
    }
    // from these deduce the spacing between bins
    binSpacing = (maxValue - minValue)/((double) nBins);
    // so you can create the histogram
    int index = 0;
    for(int i = 0; i < nValues; i++ )
    {
	index =(int) ( (x[i] - minValue) /binSpacing);
	tmpHisto[index] = tmpHisto[index] + 1;
    }
    // from this the cumulative PDF
    CDF[0] = 0.0;
    for(int i = 1; i < nBins+1; i++)
    {
	CDF[i] = CDF[i-1] + tmpHisto[i-1]/(double)nValues;
    }
   
    delete[] tmpHisto;
}

myPDF::~myPDF()
{
    delete[] CDF;
}

void myPDF::getExtents(double *_minValue, double *_maxValue, double *_binSpacing)
{
    *_minValue = minValue;
    *_maxValue = maxValue;
    *_binSpacing = binSpacing;
}

double myPDF::getCDFValue(int i)
{
    return CDF[i];
}

double myPDF::getCDFValue(double x)
{
    if( x < minValue )
    {
	return 0.0;
    }
    else if( x > (maxValue+binSpacing) )
    {
	return 1.0;
    }
    else
    {
	int index = (int) ( (x - minValue) / binSpacing );
	double lowerBound = binSpacing*(double)index + minValue;
	double interpValue = 0.0;
	if( index < nBins )
	{
	    interpValue = CDF[index] + (CDF[index+1] - CDF[index])*(x - lowerBound)/binSpacing;
	}
	else
	{
	    interpValue = CDF[index] + (1.0 - CDF[index])*(x - lowerBound)/binSpacing;
	}

	return interpValue;
    }
}

double myPDF::drawRandom(double x)
{
    return 1.0;
}
