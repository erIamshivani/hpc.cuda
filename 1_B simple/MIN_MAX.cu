#include <stdio.h>
#include <cuda.h>
#include <stdlib.h>
#include <time.h>

#define SIZE 10

__global__ void max(int *c,int *a)
{
	int i=threadIdx.x;
	*a=c[0];
	if(c[i]>*a)
	{
		*a=c[i];
	}
}

__global__ void min(int *c,int *a)
{
	int i=threadIdx.x;
	*a=c[0];
	if(c[i]<*a)
	{
		*a=c[i];
	}
}

int main(void)
{
	int i;
	srand(time(NULL));
	int a[SIZE],max_val,min_val;

	int *dev_a,*dev_max,*dev_min;

	cudaMalloc((void**)&dev_a, SIZE*sizeof(int));

	cudaMalloc((void**)&dev_max,SIZE*sizeof(int));

	cudaMalloc((void**)&dev_min,SIZE*sizeof(int));



	for(i=0;i<SIZE;i++)
	{

		a[i] = rand()%20+1;
	}

	printf("\nVector is:\n");
	for(i=0;i<SIZE;i++)
	{
		printf("%d  ",a[i]);
	}

	cudaMemcpy(dev_a,a,sizeof(a),cudaMemcpyHostToDevice);
	max<<<1,SIZE>>>(dev_a,dev_max);
	cudaMemcpy(&max_val,dev_max,sizeof(max_val),cudaMemcpyDeviceToHost);

	printf("\n\n");
    printf("The maximum value of result vector is:- %d",max_val);

    cudaMemcpy(dev_a,a,sizeof(a),cudaMemcpyHostToDevice);
    min<<<1,SIZE>>>(dev_a,dev_min);
    cudaMemcpy(&min_val,dev_min,sizeof(min_val),cudaMemcpyDeviceToHost);

    	printf("\n\n");
        printf("The minimum value of result vector is:- %d",min_val);

	return 0;
}
/*

Vector is:
7  6  7  2  18  8  17  13  16  19

The maximum value of result vector is:- 19

The minimum value of result vector is:- 2

 */
