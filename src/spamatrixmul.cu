#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <math.h>
#include <ctime>
#include <time.h>

#include <cuda_runtime.h>
#include <cublas_v2.h>

using namespace std;

#define N 256                      // define n*n matrix
#define S (N/4)                     // sparse matrix's nonzero elements per col (75% currently)
#define BLOCK_SIZE 32               // block size ( 256->16 / 1024->32 )
#define TEST_TIMES 1000             // mul module caculate times (for easy record the running time)


// simple cuda mul kernel 
static void __global__ mul_kernel(int rowsize,int colsize,int colpitch,const float *d_a,const float *d_b,float *d_c)
{
    uint index= threadIdx.x + blockIdx.x * blockDim.x;
    if(rowsize <= index) return;
    float temp=0.0f;
    for(int j=0;j<TEST_TIMES;j++)
    {
        for(int i=0;i<rowsize;i++)
        {
            temp+=d_a[i*colpitch+index]*d_b[i+j*N];
        }
        d_c[index+j*N]=temp;
        
        temp = 0.0f;
        __syncthreads();
    }
}

// cuda mul kernel with shared memory
static void __global__ mul_kernel_shared(int rowsize,int colsize,int colpitch,const float *d_a,const float *d_b,float *d_c,const int sharedsize)
{
    __shared__ float s_b[N];
    float temp=0.0f;
    uint index= threadIdx.x + blockIdx.x * blockDim.x;
    for(int j=0;j<TEST_TIMES;j++)
    {
        for(int start=0;start<rowsize;start+=sharedsize)
        {
            // load shared memory (vec)
            __syncthreads();
            for(int i=threadIdx.x;i<sharedsize&&(i+start)<rowsize;i+=blockDim.x)
                s_b[i]=d_b[i+start+j*N];
            __syncthreads();
        
            if(rowsize <= index) continue;
            int end=start+sharedsize > rowsize ? rowsize : start+sharedsize;
            for(int i=start;i<end;i++)
            {
                temp+=d_a[i*colpitch+index]*s_b[i-start];
            }
        }
        if(index<colsize)
            d_c[index+j*N]=temp;
        temp = 0;
        __syncthreads();
    }
}


// cuda mul kernel with shared memory and csr
static void __global__ mul_kernel_shared_csr(int rowsize,int colsize,int colpitch,const int *d_row,const float *d_val,const float *d_b,float *d_c,const int sharedsize)
{
    __shared__ float s_b[N];
    float temp=0.0f;
    uint index= threadIdx.x + blockIdx.x * blockDim.x;

    for(int j=0;j<TEST_TIMES;j++)
    {
        // load shared memory (vec)
        for(int start=0;start<rowsize;start+=sharedsize)
        {
            for(int i=threadIdx.x;i<sharedsize&&(i+start)<rowsize;i+=blockDim.x)
            {
                s_b[i]=d_b[i+start+j*N];
            }
            __syncthreads();
        }
    

        for(int i=0;i<S;i++)
        {
            temp+=d_val[index+N*i]*s_b[d_row[index+i*N]];
        }
    
        if(index<colsize)
            d_c[index+j*N]=temp;
        temp = 0;
        __syncthreads();
    }
        
}


// use register cache row data
static void __global__ mul_kernel_shared_csr_reg(int rowsize,int colsize,int colpitch,const int *d_row,const float *d_val,const float *d_b,float *d_c,const int sharedsize)
{
    __shared__ float s_b[N];
    float temp=0.0f;
    float val[S];
    int row[S];

    uint index= threadIdx.x + blockIdx.x * blockDim.x;

    for(int i=0;i<S;i++)
    {
        val[i]=d_val[index+N*i];
        row[i]=d_row[index+i*N];
    }

    for(int j=0;j<TEST_TIMES;j++)
    {

    // load shared memory (vec)
        for(int start=0;start<rowsize;start+=sharedsize)
        {
            for(int i=threadIdx.x;i<sharedsize&&(i+start)<rowsize;i+=blockDim.x)
            {
                s_b[i]=d_b[i+start+j*N];
            }
            __syncthreads();
        }

    
    
        for(int i=0;i<S;i++)
        {
            temp+=val[i]*s_b[row[i]];
        }
        
        if(index<colsize)
            d_c[index+j*N]=temp;
        temp = 0;
        __syncthreads();
    }
        
}

// cpu matrix mul
void mul_cpu(float *a,float *b,float *c)
{
    for(int k=0;k<TEST_TIMES;k++)
        for(int i=0;i<N;i++)
        {
            c[i+k*N]=0;
            for(int j=0;j<N;j++)
                c[i+k*N]+=(*(a+i*N+j)**(b+j+k*N));
        }
}

// test cpu and gpu mul result
bool resultcompare(float *ref,float *test,float accu)
{
    for(int i=0;i<N*TEST_TIMES;i++)
    {
        if(fabs(*(ref+i)-*(test+i))>accu) return false;
    }
    return true;
}



int main()
{
    srand(time(0));

    // Host memory

    int *sma_a_col=new int[N*S];            // CSR row array
    int *sma_a_col_tr=new int[N*S];         // CSR row array (transpose)
    float *sma_a_val=new float[N*S];        // CSR value array
    float *sma_a_val_tr=new float[N*S];     // CSR value array (transpose)

    float *matrix_a=new float[N*N];         // matrix A
    float *vec_b=new float[N*TEST_TIMES];              // vector B

    float *h_c1=new float[N*TEST_TIMES];               // Mul result C (on GPU : 1-4)
    float *h_c2=new float[N*TEST_TIMES];
    float *h_c3=new float[N*TEST_TIMES];
    float *h_c4=new float[N*TEST_TIMES];
    float *h_c5=new float[N*TEST_TIMES];

    float *ref=new float[N*TEST_TIMES];                // Mul result C (on CPU : as a reference)

    // Fill in random number (As a 75% sparse matrix)

    bool a_row[N];                          // nozero element flag
    int pos;
    for (int i = 0; i < N; i++)
    {
        for(int j=0;j<N;j++)
            a_row[j]=false;

        for(int j=0;j<S;j++)
        {
            int temp_pos = rand() % N; 
            while(a_row[temp_pos])
            {
                temp_pos++;
                if(temp_pos==N) temp_pos=0;
            } 
            a_row[temp_pos]=true;
        }

        pos=S*i;
        for(int k=0;k<N;k++)
        {
            *(matrix_a+i*N+k)=0;
            if(a_row[k])
            {
                *(sma_a_col+pos)=k;
                *(sma_a_val+pos)=rand()%10;
                *(matrix_a+i*N+k)=*(sma_a_val+pos);
                //printf("row:%d val:%f \n",*(sma_a_col+pos),*(sma_a_val+pos));
                pos++;
            }
        }
        
    }

    for (int i = 0; i < N*TEST_TIMES; i++)
        *(vec_b+i) = rand() % 10;
    /*for (int i = 0; i < N; i++)
    {
       cout << vec_b[i] << "  ";
    }*/
    

    // Cpu Mul reference 
    clock_t begin,end;
    double timer;
    begin=clock();

    for(int i=0;i<1;i++)
        mul_cpu(matrix_a,vec_b,ref);

    end=clock();
    timer=(double)(end-begin)/CLOCKS_PER_SEC;
    printf("The total cpu run time is %f ms.\n",timer*1000);


    // Matrix tranpose (for memory coalesced)

    float temp;
    for (int i = 0; i < N; i++)
        for(int j = i+1; j < N; j++)
        {
            temp = *(matrix_a+j*N+i);
            *(matrix_a+j*N+i) = *(matrix_a+i*N+j);
            *(matrix_a+i*N+j) = temp;
        }

    for (int i = 0; i < N; i++)
        for(int j = 0; j < S; j++)
        {
            *(sma_a_col_tr+j*N+i)=*(sma_a_col+i*S+j);
            *(sma_a_val_tr+j*N+i)=*(sma_a_val+i*S+j);
        }
    
    // Gpu memory malloc 

    int *d_row;
    float *d_val;
    cudaMalloc((void**)&d_row, sizeof(int)*N*S);
    cudaMalloc((void**)&d_val, sizeof(int)*N*S);

    float *d_a; 
    size_t width=N; 
    size_t height=N; 
    size_t pitch;
    cudaMallocPitch((void**)&d_a,&pitch,width*sizeof(float),height);
    pitch=N;

    float *d_b,*d_c;
    cudaMalloc((void**)&d_b, sizeof(float)*N*TEST_TIMES);
    cudaMalloc((void**)&d_c, sizeof(float)*N*TEST_TIMES);

    cudaMemcpy(d_row,sma_a_col_tr, N*S*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_val,sma_a_val_tr, N*S*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_a,matrix_a, N*N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b,vec_b, TEST_TIMES*N*sizeof(float), cudaMemcpyHostToDevice);


    //dim3 threads(BLOCK_SIZE,BLOCK_SIZE);
    uint threads=N;
    int sharedsize=threads;
    int blocknum=(N+threads-1)/threads;

    // record time & begin time
    cudaEvent_t start,stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start,0);

    // Matrix mul kernel
    for(int i=0;i<1;i++)
    {
        mul_kernel<<<blocknum, threads>>>(N,N,pitch,d_a,d_b,d_c);
        cudaThreadSynchronize();
    }

    cudaEventRecord(stop,0); 
    float costtime2;
    cudaEventSynchronize(start);    
    cudaEventSynchronize(stop); 
    cudaEventElapsedTime(&costtime2,start,stop);
    printf("The total gpu run time is %f ms.\n",costtime2);

    cudaMemcpy(h_c2, d_c,TEST_TIMES*N*sizeof(float), cudaMemcpyDeviceToHost);


    // Matrix mul using shared memory kernel
    cudaEventRecord(start,0);

    for(int i=0;i<1;i++)
    {
        mul_kernel_shared<<<blocknum, threads>>>(N,N,pitch,d_a,d_b,d_c,sharedsize);
        cudaThreadSynchronize();
    }

    cudaEventRecord(stop,0);  
    float costtime;
    cudaEventSynchronize(start);    
    cudaEventSynchronize(stop); 
    cudaEventElapsedTime(&costtime,start,stop);
    printf("The total gpu (use shared memory) run time is %f ms.\n",costtime);

    cudaMemcpy(h_c1, d_c,TEST_TIMES*N*sizeof(float), cudaMemcpyDeviceToHost);

    // Matrix mul using shared memory and csr kernel

    cudaEventRecord(start,0);

    for(int i=0;i<1;i++)
    {
        mul_kernel_shared_csr<<<blocknum, threads>>>(N,N,pitch,d_row,d_val,d_b,d_c,sharedsize);
        cudaThreadSynchronize();
    }

    cudaEventRecord(stop,0); 
    float costtime3;
    cudaEventSynchronize(start);    
    cudaEventSynchronize(stop); 
    cudaEventElapsedTime(&costtime3,start,stop);
    printf("The total gpu (using csr and shared memory) run time is %f ms.\n",costtime3);
    
    cudaMemcpy(h_c3, d_c,TEST_TIMES*N*sizeof(float), cudaMemcpyDeviceToHost);

    // use register
    cudaEventRecord(start,0);

    for(int i=0;i<1;i++)
    {
        mul_kernel_shared_csr_reg<<<blocknum, threads>>>(N,N,pitch,d_row,d_val,d_b,d_c,sharedsize);
        cudaThreadSynchronize();
    }

    cudaEventRecord(stop,0); 
    float costtime5;
    cudaEventSynchronize(start);    
    cudaEventSynchronize(stop); 
    cudaEventElapsedTime(&costtime5,start,stop);
    printf("The total gpu (using csr by register and shared memory) run time is %f ms.\n",costtime5);
    
    cudaMemcpy(h_c5, d_c,TEST_TIMES*N*sizeof(float), cudaMemcpyDeviceToHost);

    // Matrix using cublas function call
    float alpha = 1;
    float beta = 0;
    int M=1;                // B->vector

    cublasHandle_t handle;
    cublasCreate(&handle);

    cudaEventRecord(start,0);

    // matrix cublas call
    for(int i=0;i<1000;i++)
    {
        cublasSgemm(handle,
        CUBLAS_OP_T,  
        CUBLAS_OP_N,   
        M,                    // row of B
        N,                    // col of A
        N,                    // row of B
        &alpha,           
        d_b,            
        M,                    
        d_a,         
        N,         
        &beta,          
        d_c,           
        M);
        cudaThreadSynchronize();

    }

    cudaEventRecord(stop,0); 
    float costtime4;
    cudaEventSynchronize(start);    
    cudaEventSynchronize(stop); 
    cudaEventElapsedTime(&costtime4,start,stop);
    printf("The total gpu (using cublas) run time is %f ms.\n",costtime4);
    
    cudaMemcpy(h_c4, d_c,N*sizeof(float), cudaMemcpyDeviceToHost);


    // Correct test
    printf("Test correct:\n");
    bool res;
    res=resultcompare(ref,h_c1,1e-4f);
    if(res) printf("1PASSED!\n");
    else printf("1FAILED!\n");

    res=resultcompare(ref,h_c2,1e-4f);
    if(res) printf("2PASSED!\n");
    else printf("2FAILED!\n");

    res=resultcompare(ref,h_c3,1e-4f);
    if(res) printf("3PASSED!\n");
    else printf("3FAILED!\n");

    res=resultcompare(ref,h_c4,1e-4f);
    if(res) printf("4PASSED!\n");
    else printf("4FAILED!\n");

    res=resultcompare(ref,h_c5,1e-4f);
    if(res) printf("5PASSED!\n");
    else printf("5FAILED!\n");

    // test diff
    /*for(int i=0;i<TEST_TIMES*N;i++)
    {
        printf("c=%f\to1=%f\to2=%f\to3=%f\to4=%f\to5=%f\n",*(ref+i),*(h_c1+i),*(h_c2+i),*(h_c3+i),*(h_c4+i),*(h_c5+i));
    }*/


    // Free memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    free(sma_a_col);
    free(sma_a_val);
    free(matrix_a);
    free(vec_b);
    free(h_c1);
    free(h_c2);
    free(h_c3);
    free(h_c4);
    free(ref);
    return 0;
}