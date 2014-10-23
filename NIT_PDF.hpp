/*
The MIT License (MIT)

Copyright (c) 2014 Mathias Helsen

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#ifndef _NIT_PDF_HPP
#define _NIT_PDF_HPP

#include <iostream>
#include <math.h>
#include <vector>

#include "Generic_PDF.hpp"

using namespace std;

class NIT_PDF : public Generic_PDF{
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
	NIT_PDF(int _nBins) { nBins = _nBins; };
	~NIT_PDF();

	double getCDFValue(double x);
	double getPDFValue(double x);
	double getPDF(vector<double> *ranges, vector<double> *values);
	double getAverage();
	double getStandardDev() { return std; };
	void generatePDF( std::vector<double> *x );
	void setDrift( double drift );
	double drawRandom(double x);
};

#endif
