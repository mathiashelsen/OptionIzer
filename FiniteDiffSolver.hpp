#ifndef _FINITEDIFFSOLVER_HPP
#define _FINITEDIFFSOLVER_HPP

#include "Solver.hpp"
#include "VanillaOption.hpp"

#include <gsl/gsl_vector.h>
#include <gsl/gsl_linalg.h>
#include <math.h>
#include <algorithm>

class FiniteDiffSolver : public Solver<VanillaOption>
{
    private:
	int Nz;
	int Nt;
	gsl_vector *f;
	gsl_vector *g;
	gsl_vector *diag;
	gsl_vector *super;
	gsl_vector *sub;
    public:
	void operator()(VanillaOption *option);
	FiniteDiffSolver( int _Nz, int _Nt );
	~FiniteDiffSolver();
};

#endif
