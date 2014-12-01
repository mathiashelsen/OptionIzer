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

#ifndef _BINOMIAL_HPP
#define _BINOMIAL_HPP

#include <iostream>
#include <math.h>
#include <algorithm>

class Binomial
{
    private:
	double S0;
	double K;
	double sigma;
	double r;
	double T;
	int N;

	double dt;

	double u, d, p;
	double price, delta, gamma, vega, rho, theta;
	void recalc(void);

    public:
	Binomial(double _underlying,
	    double _strike,
	    double _volatility,
	    double _riskless,
	    double _T,
	    int _N );
	~Binomial() {};

	void setS0(double _S0){ S0 = _S0; };
	void setK(double _K){ K = _K; };
	void setSigma(double _sigma){ sigma = _sigma; recalc(); };
	void setT(double _T){ T = _T; recalc(); };
	void setN(int _N){ N = _N; recalc(); };

	void evaluate();	
	void calcPrice(double *put);
	void calcDelta(double *deltaPut) { *deltaPut = delta; };
	void calcGamma(double *gammaPut) { *gammaPut = gamma; };
	void calcVega(double *vegaPut) { *vegaPut = vega ; };
	void calcTheta(double *thetaPut) { *thetaPut = theta; };
	void calcRho(double *rhoPut) { *rhoPut = rho; };
};

#endif
