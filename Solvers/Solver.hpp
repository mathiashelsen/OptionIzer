#ifndef _SOLVER_HPP
#define _SOLVER_HPP

template<class OptionType> class Solver
{
    public:
	virtual void operator()(OptionType *option) {return;};
	virtual void init() {return;};
};

#endif
