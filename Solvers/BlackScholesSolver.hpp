#ifndef _BLACKSCHOLESSOLVER_HPP
#define _BLACKSCHOLESSOLVER_HPP

#include "Solver.hpp"
#include "../OptionTypes/EuroOption.hpp"

#include <iostream>
#include <math.h>
#include <algorithm>

#define SQRT2 1.4142135623730950488
#define SQRT2PI 2.50662827463100050242

template<class OptionType> class BlackScholesSolver: public Solver<OptionType>
{
    public:
	~BlackScholesSolver(){ return; }; 
	void operator()(OptionType *option);
	void init();
};

template<> class BlackScholesSolver<EuroOption>: public Solver<EuroOption>
{
    public:
	void operator()(EuroOption *option);
	void init() { return; };
	BlackScholesSolver();
	~BlackScholesSolver();
};

#endif
