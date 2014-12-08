#ifndef _BINOMIALSOLVER_HPP
#define _BINOMIALSOLVER_HPP

#include "Solver.hpp"
#include "VanillaOption.hpp"

#include <iostream>
#include <math.h>
#include <algorithm>

class BinomialSolver : private Solver<VanillaOption>
{
    private:
	int N;
	double **assetValues;
	double **optionValues;

	double price, delta, gamma, theta;
    public:
	void operator()(VanillaOption *option);
	BinomialSolver(int _N);
	~BinomialSolver();
};

#endif
