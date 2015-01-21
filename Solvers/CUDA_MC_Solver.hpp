#ifndef _CUDA_MC_SOLVER_HPP
#define _CUDA_MC_SOLVER_HPP

#include "../OptionTypes/EuroOption.hpp"
#include "../OptionTypes/AsianOption.hpp"

#include <assert.h>
#include <iostream>
#include <stdio.h>

#include <cuda.h>
#include <curand.h>

template<class OptionType> class CUDA_MC_Solver : public Solver<OptionType>
{
    public:
	~CUDA_MC_Solver(){ return; }; 
	void operator()(OptionType *option);
	void init();
	void free();
};

template<> class CUDA_MC_Solver<EuroOption> : public Solver<EuroOption>
{
    private:
	int Nseries, Nsteps;
	float *returns, *assets, *payoffs;
	curandGenerator_t gen;
    public:
	CUDA_MC_Solver(int _NSeries, int _NStep);
	~CUDA_MC_Solver();
	void operator()(EuroOption *option);
	void init();
	void free();
};

template<> class CUDA_MC_Solver<AsianOption> : public Solver<AsianOption>
{
    private:
	int Nseries, Nsteps;
	float *returns, *assets, *payoffs;
	curandGenerator_t gen;
    public:
	CUDA_MC_Solver(int _NSeries, int _NStep);
	~CUDA_MC_Solver();
	void operator()(AsianOption *option);
	void init();
	void free();
};

#endif
