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

#ifndef _BLACKSCHOLES_HPP
#define _BLACKSCHOLES_HPP

#include <iostream>
#include <math.h>

#define SQRT2 1.4142135623730950488
#define SQRT2PI 2.50662827463100050242

class BlackScholes
{
    private:
	double S0;
	double K;
	double sigma;
	double r;
	double T;
	double d1;
	double d2;

	void recalcd();

    public:
	BlackScholes(double _underlying,
	    double _strike,
	    double _volatility,
	    double _riskless,
	    double _T );
	~BlackScholes() {};

	void setS0(double _S0){ S0 = _S0; recalcd(); };
	void setK(double _K){ K = _K; recalcd(); };
	void setSigma(double _sigma){ sigma = _sigma; recalcd(); };
	void setT(double _T){ T = _T; recalcd(); };
	
	void calcPrice(double *call, double *put);
	void calcIVCall(double callPrice, double *callIV);
	void calcIVPut(double putPrice, double *putIV);

	void calcDelta(double *deltaCall, double *deltaPut);
	void calcGamma(double *gammaCall, double *gammaPut);
	void calcVega(double *vegaCall, double *vegaPut);
	void calcTheta(double *thetaCall, double *thetaPut);
	void calcRho(double *rhoCall, double *rhoPut);
};

#endif
