#ifndef _LOOKBACKOPTION_HPP
#define _LOOKBACKOPTION_HPP

#include "Option.hpp"

enum LookBackType
{
    LookBackTypeMin, LookBackTypeMax
};


class LookBackOption : public Option
{
    public:
	LookBackOption(double _S0, double _K, double _sigma, double _r, double _T, bool _put, LookBackType _type);
	// Inputs
	double sigma;
	double T;
	bool put;
	double K;
	LookBackType type;
};

#endif
