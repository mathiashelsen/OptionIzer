#ifndef _MYPDF_HPP
#define _MYPDF_HPP

#include <iostream>
#include <math.h>

class myPDF{
    private:
	int nBins;
	double binSpacing;
	double minValue;
	double maxValue;
	double *CDF;
    public:
	void getExtents(double *_minValue, double *_maxValue, double *_binSpacing);
	double getCDFValue(int i);
	double getCDFValue(double x);
	myPDF(int nBins, double *x, int nValues);
	~myPDF();
	double drawRandom(double x);
};

#endif
