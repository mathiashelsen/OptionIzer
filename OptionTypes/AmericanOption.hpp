#ifndef _AMERICANOPTION_HPP
#define _AMERICANOPTION_HPP

#include "VanillaOption.hpp"

class AmericanOption : public VanillaOption
{
    public:
	AmericanOption(double _S0, double _K, double _sigma, double _r, double _T, bool _put) : 
	    VanillaOption(_S0, _K, _sigma, _r, _T, _put)
	    { return; };
};

#endif
