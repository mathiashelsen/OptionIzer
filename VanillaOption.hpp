#ifndef _VANILLAOPTION_HPP
#define _VANILLAOPTION_HPP

#include "Solver.hpp"

class VanillaOption
{
    friend class BinomialSolver;
    protected:
	double S0;
	double K;
	double sigma;
	double r;
	double T;
	bool american;
    public:
	VanillaOption();
	VanillaOption(double _S0, double _K, double _sigma, double _r, double _T, bool _american);
	~VanillaOption();
};

#endif
