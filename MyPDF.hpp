#ifndef _MYPDF_HPP
#define _MYPDF_HPP

#include <iostream>
#include <math.h>
#include <vector>

class MyPDF{
    private:
	int nBins;
	double binSpacing;
	double minValue;
	double maxValue;
	double *CDF;
	double *PDF;
    public:
	void getExtents(double *_minValue, double *_maxValue, double *_binSpacing);
	double getCDFValue(double x);
	double getPDFValue(double x);
	double getPDF(vector<double> *ranges, vector<double> *values);
	MyPDF(int nBins, std::vector<double> *x);
	~MyPDF();
	double drawRandom(double x);
};

#endif
