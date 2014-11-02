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
    strike = _strike;
    volatility = _volatility;
    riskless = _riskless;
    T = _T;
}

void BlackScholes::evaluate(double *call, double *put)
{
    double d1 = 0.0;
    double d2 = 0.0;
    d1 = (log(underlying/strike) + (riskless + 0.5*volatility*volatility)*T)/(volatility*sqrt(T));
    d2 = d1 - volatility*sqrt(T);

    *call = NCDF(0.0, 1.0, d1)*underlying - NCDF(0.0, 1.0, d2)*strike*exp(-T*riskless);
    *put = strike*exp(-T*riskless) - underlying + *call;
}
