#ifndef _BLACKSCHOLESSOLVER_HPP
#define _BLACKSCHOLESSOLVER_HPP

#include "Solver.hpp"
#include "VanillaOption.hpp"

#include <iostream>
#include <math.h>
#include <algorithm>

#define SQRT2 1.4142135623730950488
#define SQRT2PI 2.50662827463100050242

class BlackScholesSolver: public Solver<VanillaOption>
{
    public:
	void operator()(VanillaOption *option);
	BlackScholesSolver();
	~BlackScholesSolver();
};

#endif
