#include "CUDA_MC_Solver.hpp"
#include "CUDA_MC_LookBack_Solver.cuh"

CUDA_MC_Solver<LookBackOption>::CUDA_MC_Solver(int _Nseries, int _Nsteps)
{
    Nseries = _Nseries;
    Nsteps = _Nsteps; 

    returns = NULL;
    assets = NULL;
    payoffs = NULL;
}

CUDA_MC_Solver<LookBackOption>::~CUDA_MC_Solver()
{
    curandDestroyGenerator( gen );

    if( returns )
	cudaFree( returns );
    if( assets )
	cudaFree( assets );
    if( payoffs )
	cudaFree( payoffs );
};

void CUDA_MC_Solver<LookBackOption>::operator()(LookBackOption *option)
{
    assert( returns );
    assert( assets );
    assert( payoffs );
    float *localPayoffs = new float[Nseries];
    assert( localPayoffs );
    float scale = (float)option->T/(float)Nsteps;

    float call = 1.0;
    if(option->put)
	call = -1.0;

    int threadsPerBlock = 256;
    int nBlocks = Nseries/threadsPerBlock;
    LookBackKernel<option->type><<<nBlocks, threadsPerBlock>>>(returns, 
					    assets, 
					    payoffs, 
					    (float)option->r, 
					    (float)option->S0, 
					    (float)option->K, 
					    (float) option->sigma*scale, 
					    (float) option->T,
					    call,
					    Nseries, 
					    Nsteps);
    
    cudaThreadSynchronize();
    cudaMemcpy( (void *) localPayoffs, payoffs, sizeof(float)*Nseries, cudaMemcpyDeviceToHost );

    // This ends all the calls to cuda, now just averaging over the payoffs
    double avg;
    for(int i = 0; i < Nseries; i++)
    {
	avg += (double) localPayoffs[i];
    }
    avg /= (double) Nseries;
    option->price = avg;
    delete[] localPayoffs;
};

void CUDA_MC_Solver<LookBackOption>::init()
{
    assert( cudaMalloc( (void **) &returns, Nseries*Nsteps*sizeof(float) ) == cudaSuccess);
    assert( cudaMalloc( (void **) &assets, Nseries*sizeof(float) ) == cudaSuccess);
    assert( cudaMalloc( (void **) &payoffs, Nseries*sizeof(float) ) == cudaSuccess);

    assert( curandCreateGenerator( &gen, CURAND_RNG_PSEUDO_MTGP32 ) == CURAND_STATUS_SUCCESS);
    assert( curandSetPseudoRandomGeneratorSeed( gen, 1234ULL ) == CURAND_STATUS_SUCCESS);
    assert( curandGenerateNormal( gen, returns, Nseries*Nsteps, 0.0, 1.0 ) == CURAND_STATUS_SUCCESS);
};

void CUDA_MC_Solver<LookBackOption>::free()
{
    curandDestroyGenerator( gen );

    if( returns )
	cudaFree( returns );
	returns = NULL;
    if( assets )
	cudaFree( assets );
	assets = NULL;
    if( payoffs )
	cudaFree( payoffs );
	payoffs = NULL;
};
