#include<cuda.h>
#include<stdio.h>

int main(void) {
    void MatrixMultiplication(float *, float *, float *, int);
    const int Width = 3;
    float M[Width*Width], N[Width*Width], P[Width*Width];
    for(int i = 0; i < (Width*Width) ; i++) {
        M[i] = rand()%10;
        N[i] = rand()%10;
        P[i] = 0;
    }
    printf("First Matrix:\n");
    for(int i = 0; i < (Width*Width) ; i=i+Width) {
    for(int j = 0; j < (Width) ; j++) {
            printf("%f \t", M[i+j]);
        }
    printf("\n");
    }
    printf("\n");
    printf("Second Matrix:\n");
    for(int i = 0; i < (Width*Width) ; i=i+Width) {
        for(int j = 0; j < (Width) ; j++) {
                printf("%f \t", N[i+j]);
            }
        printf("\n");
        }
    printf("\n");
    printf("Multiplication :\n");
    MatrixMultiplication(M, N, P, Width);
    for(int i = 0; i < (Width*Width) ; i=i+Width) {
        for(int j = 0; j < (Width) ; j++) {
                printf("%f \t", P[i+j]);
            }
        printf("\n");
        }
    int quit;
    scanf("%d",&quit);
    return 0;
}

//Matrix multiplication kernel - thread specification
__global__ void MatrixMulKernel(float *Md, float *Nd, float *Pd, int Width) {
    //2D Thread ID
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    //Pvalue stores the Pd element that is computed by the thread
    float Pvalue = 0;

    for(int k = 0; k < Width ; ++k) {
        float Mdelement = Md[ty*Width + k];
        float Ndelement = Nd[k*Width + tx];
        Pvalue += (Mdelement*Ndelement);
    }

    Pd[ty*Width + tx] = Pvalue;
}

void MatrixMultiplication(float *M, float *N, float *P, int Width) {
    int size = Width*Width*sizeof(float);
    float *Md, *Nd, *Pd;

    //Transfer M and N to device memory
    cudaMalloc((void**)&Md, size);
    cudaMemcpy(Md,M,size,cudaMemcpyHostToDevice);
    cudaMalloc((void**)&Nd, size);
    cudaMemcpy(Nd,N,size,cudaMemcpyHostToDevice);

    //Allocate P on the device
    cudaMalloc((void**)&Pd,size);

    //Setup the execution configuration
    dim3 dimBlock(Width,Width);
    dim3 dimGrid(1,1);

    //Launch the device computation threads!
    MatrixMulKernel<<<dimGrid,dimBlock>>>(Md,Nd,Pd,Width);

    //Transfer P from device to host
    cudaMemcpy(P,Pd,size,cudaMemcpyDeviceToHost);

    //Free device matrices
    cudaFree(Md);
    cudaFree(Nd);
    cudaFree(Pd);
}
