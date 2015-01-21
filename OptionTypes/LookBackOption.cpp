#include "LookBackOption.hpp"

LookBackOption::LookBackOption(double _S0, double _sigma, double _r, double _T, bool _put, bool _min)
{
    S0 = _S0;
    sigma = _sigma;
    r = _r;
    T = _T;
    put = _put;
    min = _min;
}
