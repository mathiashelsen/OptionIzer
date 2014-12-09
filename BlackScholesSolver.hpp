#ifndef _BLACKSCHOLESSOLVER_HPP
#define _BLACKSCHOLESSOLVER_HPP

#include "Solver.hpp"
#include "VanillaOption.hpp"

#include <iostream>
#include <math.h>
#include <algorithm>

static double NCDF(double mu, double sigma, double x)
{
    return (1.0 + erf((x-mu)/(sigma*SQRT2)))/2.0;
}

class BlackScholesSolver: public Solver<VanillaOption>
{

};

#endif
