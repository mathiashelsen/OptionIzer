#include <iostream>

class myPDF{
    private:
	double binSpacing;
	double *binValues;
    public:
	myPDF(int nBins, double *x, int nValues);
	~myPDF();
	double drawRandom(double x);
};
