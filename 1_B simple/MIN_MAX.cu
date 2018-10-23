#include<stdio.h>
#include<cuda.h>
#include<time.h>

#define N 8
__global__ void minimum(int *arr,int *minVal)
{
    int tid = threadIdx.x;
    
    *minVal = 999;
    atomicMin(minVal,arr[tid]);
}

__global__ void maximum(int *arr, int *maxVal)
{
    int tid = threadIdx.x;
    
    *maxVal  = 0;
    atomicMax(maxVal,arr[tid]);
}

__global__ void add(int *arr,int *avg) //only sums all the elements of the array
{
    int tid = threadIdx.x;
    *avg=0;
    atomicAdd(avg,arr[tid]);
}

int main(void)
{
    srand(time(NULL));
    int i;
    int arr[N],minVal,maxVal,addVal;
    
    int *dev_arr,*dev_min,*dev_max,*dev_add;
    
    cudaMalloc((void **)&dev_arr, N*sizeof(int));
    cudaMalloc((void **)&dev_min, N*sizeof(int));
    cudaMalloc((void **)&dev_max, N*sizeof(int));
    cudaMalloc((void **)&dev_add, N*sizeof(int));
    
    for(i=0;i<N;i++)
    {
        arr[i] = rand()%20 + 1;
    }
    
    printf("The array is : ");
   
    for(i=0;i<N;i++)
    {
        printf(" %d ",arr[i]);
    }
    
    cudaMemcpy(dev_arr,arr,sizeof(arr),cudaMemcpyHostToDevice);
    
    minimum<<<1,N>>>(dev_arr,dev_min);
    
    cudaMemcpy(&minVal,dev_min,sizeof(minVal),cudaMemcpyDeviceToHost);
    
    printf("The minimum value is : %d",minVal);
    
    
    
    cudaMemcpy(dev_arr,arr,sizeof(arr),cudaMemcpyHostToDevice);
    
    maximum<<<1,N>>>(dev_arr,dev_max);
    
    cudaMemcpy(&maxVal,dev_max,sizeof(maxVal),cudaMemcpyDeviceToHost);
    
    printf("The maximum value is : %d",maxVal);
    
    
    cudaMemcpy(dev_arr,arr,sizeof(arr),cudaMemcpyHostToDevice);
    
    add<<<1,N>>>(dev_arr,dev_add);
    
    cudaMemcpy(&addVal,dev_add,sizeof(addVal),cudaMemcpyDeviceToHost);
    
    printf("The average value is : %d",addVal/N);
    
    return 0;
}
