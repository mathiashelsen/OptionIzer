#include "BlackScholesSolver.hpp"

static inline double NCDF(double mu, double sigma, double x)
{
    return (1.0 + erf((x-mu)/(sigma*SQRT2)))/2.0;
}

