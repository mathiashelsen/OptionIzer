#include "myPDF.hpp"

myPDF::myPDF(int _nBins, double *x, int nValues)
{
    nBins = _nBins;
    // Local variables to be freed after initializing
    double * tmpHisto = new double[nBins];
    maxValue = x[0], minValue = x[0];
    // Allocating some memory for the inverse CDF
    CDF = new double[nBins];
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
    CDF[0] = tmpHisto[0] / (double)nValues;
    for(int i = 1; i < nBins; i++)
    {
	CDF[i] = CDF[i-1] + tmpHisto[i]/(double)nValues;
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
	int index = ((int) ( (x - minValue) / binSpacing )) ;
	double interpValue = 0.0;
	if( index < nBins )
	{
	    interpValue = CDF[index-1] + (CDF[index] - CDF[index-1])*(x - binSpacing*(double)index)/binSpacing;
	}
	else
	{
	    interpValue = CDF[index-1] + (1.0 - CDF[index-1])*(x - binSpacing*(double)index)/binSpacing;
	}

	return interpValue;
	//return CDF[index];
	//return CDF[(int)((x-minValue)/binSpacing)];
    }
    /*
    double interpVal = 0.0;
    
    if( (x < minValue) && (x >= (minValue - binSpacing)) )
    {
	interpVal = x*CDF[0]/binSpacing;
    }
    else if( (x > maxValue) && (x <= (maxValue + binSpacing)) )
    {
	interpVal = CDF[nBins-1] + (1.0 - CDF[nBins-1])*(x - maxValue)/binSpacing;
    }
    else if( (x > (maxValue + binSpacing )) )
    {
	interpVal = 1.0;
    }
    else
    {
	int index = (int) ( (x - minValue)/binSpacing );
	double lowerBound = minValue + binSpacing*(double)index;
	interpVal = CDF[index] + (CDF[index+1] - CDF[index])*(x - lowerBound)/binSpacing;
    }
   
    return interpVal;
    */
}

double myPDF::drawRandom(double x)
{
    return 1.0;
}
