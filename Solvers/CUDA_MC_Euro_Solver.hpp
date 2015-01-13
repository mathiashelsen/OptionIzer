#ifndef _CUDA_MC_EURO_SOLVER
#define _CUDA_MC_EURO_SOLVER

#include "../OptionTypes/VanillaOption.hpp"

#include <cuda.h>
#include <curand.h>

class CUDA_MC_Euro_Solver : public Solver<VanillaOption>
{
private:
    int Nseries;
    int Nsteps;
public:
    CUDA_MC_Euro_Solver(int _NSeries, int _NStep);
    void operator()(VanillaOption *option);
    
};

#endif
