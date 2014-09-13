#ifndef _MYPDF_HPP
#define _MYPDF_HPP

#include <iostream>
#include <math.h>
#include <vector>

using namespace std;

class MyPDF{
    private:
	int nBins;
	double binSpacing;
	double minValue;
	double maxValue;
	double *CDF;
	double *PDF;
	double avg;
	double std;
    public:
	double getCDFValue(double x);
	double getPDFValue(double x);
	double getPDF(vector<double> *ranges, vector<double> *values);
	double getAverage();
	double getStandardDev() { return std; };
	void generatePDF( std::vector<double> *x);
	MyPDF(int _nBins) { nBins = _nBins; };
	MyPDF(int nBins, std::vector<double> *x);
	~MyPDF();
	double drawRandom(double x);
};

#endif
