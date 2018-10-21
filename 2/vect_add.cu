#include <stdio.h>
#include<cuda.h>
#include <stdlib.h>
#include<time.h>

#define SIZE 20

__global__ void vectsum(int *x,int *y,int *z)
{
	int tid=blockIdx.x*blockDim.x+threadIdx.x;
	z[tid]=x[tid]+y[tid];
}

int main(void)
{
	int i;
	// srand(time(NULL));
	int a[SIZE],b[SIZE],c[SIZE];

	int *dev_a,*dev_b,*dev_c;

	cudaMalloc((void **)&dev_a, SIZE*sizeof(int));
	cudaMalloc((void **)&dev_b, SIZE*sizeof(int));
	cudaMalloc((void **)&dev_c, SIZE*sizeof(int));

	for(i=0;i<SIZE;i++)
	{
		a[i] = rand()%20+1;
	}

	printf("\nThe 1st vector is:\n");
	for(i=0;i<SIZE;i++)
	{
		printf("%d  ",a[i]);
	}

	for(i=0;i<SIZE;i++)
	{
		b[i] = rand()%20+1;
	}

	printf("\nThe 2nd vector is:\n");
	for(i=0;i<SIZE;i++)
	{
		printf("%d  ",b[i]);
	}

	cudaMemcpy(dev_a,a,sizeof(a),cudaMemcpyHostToDevice);
	cudaMemcpy(dev_b,b,sizeof(b),cudaMemcpyHostToDevice);
	vectsum<<<1,SIZE>>>(dev_a,dev_b,dev_c);
	cudaMemcpy(&c,dev_c,sizeof(c),cudaMemcpyDeviceToHost);

	printf("\nThe result is:\n");
	for(int i=0;i<SIZE;i++)
	{
		printf("%d  ",c[i]);
	}


	return 0;
}
