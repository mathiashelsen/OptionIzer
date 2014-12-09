#include "VanillaOption.hpp"

VanillaOption::VanillaOption(double _S0, double _K, double _sigma, double _r, double _T, bool _american, bool _put)
{
    S0 = _S0;
    K = _K;
    sigma = _sigma;
    r = _r;
    T = _T;
    american = _american; 
    put = _put;
};

VanillaOption::~VanillaOption()
{
    
};
