#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>
#include <numeric>
#define SIZE 16
using namespace std;

__global__ void sum(int* input)
{
	const int tid = threadIdx.x;

	int step_size = 1;
	int number_of_threads = blockDim.x;

	while (number_of_threads > 0)
	{
		if (tid < number_of_threads) // still alive?
		{
			const int fst = tid * step_size * 2;
			const int snd = fst + step_size;
			input[fst] += input[snd];
		}

		step_size <<= 1;
		number_of_threads >>= 1;
	}
}

int main()
{
	int count = SIZE;
	const int size = count*sizeof(int);
	int h[SIZE];
	for(int i=0;i<count;i++)
	{
	h[i]=rand%20+1;
	}
	for(int i=0;i<count;i++)
	{
	printf("%d ",h[i]);
	}

	int* d;

	cudaMalloc(&d, size);
	cudaMemcpy(d, h, size, cudaMemcpyHostToDevice);

	sum <<<1, count>>>(d);

	int result;
	cudaMemcpy(&result, d, sizeof(int), cudaMemcpyDeviceToHost);

	cout << "Sum is " << result << endl;

	//getchar();

	cudaFree(d);
	

	return 0;
}

