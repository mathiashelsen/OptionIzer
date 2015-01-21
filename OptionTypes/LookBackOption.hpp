#ifndef _LOOKBACKOPTION_HPP
#define _LOOKBACKOPTION_HPP

#include "Option.hpp"

class LookBackOption : public Option
{
    public:
	LookBackOption(double _S0, double _sigma, double _r, double _T, bool _put, bool _min);
	// Inputs
	double sigma;
	double T;
	bool put;
	bool min;
};

#endif
