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

#include "BlackScholes.hpp"

static double NCDF(double mu, double sigma, double x)
{
    return (1.0 + erf((x-mu)/(sigma*SQRT2)))/2.0;
}

BlackScholes::BlackScholes(double _underlying,
	    double _strike,
	    double _volatility,
	    double _riskless,
	    double _T)
{
    S0 = _underlying;
    K = _strike;
    sigma = _volatility;
    r = _riskless;
    T = _T;

    d1 = (log(S0/K) + (r + 0.5*sigma*sigma)*T)/(sigma*sqrt(T));
    d2 = d1 - sigma*sqrt(T);
}

void BlackScholes::recalcd()
{
    d1 = (log(S0/K) + (r + 0.5*sigma*sigma)*T)/(sigma*sqrt(T));
    d2 = d1 - sigma*sqrt(T);
}

void BlackScholes::calcPrice(double *call, double *put)
{
    *call = NCDF(0.0, 1.0, d1)*S0 - NCDF(0.0, 1.0, d2)*K*exp(-T*r);
    *put = K*exp(-T*r) - S0 + *call;
}

void BlackScholes::calcIVCall(double callPrice, double *callIV)
{
    //sigma = 0.2;
    double sigmaErr = 1.0;
    double fn, dfn;
    double d1 = 0.0;
    double d2 = 0.0;
    while(fabs(sigmaErr) > 1.0e-6)
    {	
	recalcd();
	fn = NCDF(0.0, 1.0, d1)*S0 - NCDF(0.0, 1.0, d2)*K*exp(-T*r) - callPrice;	
	dfn = S0*sqrt(T)*exp(-d1*d1*0.5)/SQRT2PI;
	sigmaErr = fn/dfn;
	sigma += -sigmaErr;
    }
    std::cout << std::endl;
    *callIV = sigma;
}

void BlackScholes::calcDelta(double *deltaCall, double *deltaPut)
{
    *deltaCall = NCDF(0.0, 1.0, d1);
    *deltaPut = *deltaCall - 1.0;
}

void BlackScholes::calcGamma(double *gammaCall, double *gammaPut)
{
    *gammaCall = exp(-d1*d1*0.5)/(S0*sigma*sqrt(T)*SQRT2PI);
    *gammaPut = *gammaCall;
}

void BlackScholes::calcVega(double *vegaCall, double *vegaPut)
{
    *vegaCall = S0*sqrt(T)*exp(-d1*d1*0.5)/SQRT2PI;
    *vegaPut = *vegaCall;
}

void BlackScholes::calcTheta( double *thetaCall, double *thetaPut)
{
    *thetaCall = -S0*sigma*exp(-d1*d1*0.5)/(SQRT2PI*2.0*sqrt(T)) - r*K*exp(-r*T)*NCDF(0.0, 1.0, d2);
    *thetaPut = -S0*sigma*exp(-d1*d1*0.5)/(SQRT2PI*2.0*sqrt(T)) + r*K*exp(-r*T)*NCDF(0.0, 1.0, -d2);
}
