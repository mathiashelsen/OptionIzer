#ifndef _VANILLACALL_HPP
#define _VANILLACALL_HPP

#include "Option.hpp"

class VanillaOption : public Option
{
    protected:
	bool call;
	double strike;
};

#endif
