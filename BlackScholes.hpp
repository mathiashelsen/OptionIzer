#ifndef _BLACKSCHOLES_HPP
#define _BLACKSCHOLES_HPP

#include <iostream>
#include <math.h>

#define SQRT2 1.4142135623730950488

class BlackScholes
{
    private:
	double underlying;
	double strike;
	double volatility;
	double riskless;
	double T;

    public:
	BlackScholes(double _underlying,
	    double _strike,
	    double _volatility,
	    double _riskless,
	    double _T);
	~BlackScholes() {};
	void evaluate(double *call, double *put);

};

#endif
