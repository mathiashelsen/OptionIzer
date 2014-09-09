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
    double maxErr = 0.0001;
    /*
    std::cout << x << "\n";
    if( (x > CDF[1]) && (x < CDF[nBins] ) )
    {
	double x_0 = minValue;
	double fx_0 = getCDFValue(x_0) - x;
	double x_1 = maxValue;
	double fx_1 = getCDFValue(x_1) - x;
	double x_2 = x_1 - fx_1*((x_1 - x_0)/(fx_1 - fx_0));
	double fx_2 = getCDFValue(x_2) - x;
	while(fabs(fx_2) > maxErr)
	{
	    x_0 = x_1;
	    fx_0 = fx_1;
	    x_1 = x_2;
	    fx_1 = fx_2;	
	    x_2 = x_1 - fx_1*((x_1 - x_0)/(fx_1 - fx_0));
	    fx_2 = getCDFValue(x_2) - x;
	    std::cout << x_2 << "\t" << fx_2 << "\n";
	}
	return x_2;
    }
    */
    if( (x >= CDF[1]) && (x <= CDF[nBins]) )
    {
	double l = minValue;
	double u = maxValue;
	double c = (l+u)*0.5;
	double fc = getCDFValue(c) - x;
	while( fabs(fc) > maxErr )
	{
	    if( fc < 0.0 )
	    {
		l = c;
	    }
	    else
	    {
		u = c;
	    }
	    c = (l+u)*0.5;
	    fc = getCDFValue(c) - x;
	}
	return c;	
    }
    else if( x < CDF[1] )
    {
	double interp = minValue - binSpacing + binSpacing*x/CDF[1];
	return interp;
    }
    else
    {
	double interp = maxValue + binSpacing*(x - CDF[nBins])/(1.0 - CDF[nBins]);
	return interp;
    }
}
