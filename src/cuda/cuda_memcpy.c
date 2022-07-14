#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <assert.h>
#include <string.h>
#include <cuda_runtime_api.h>

#define NUM_SIZES 27

int cuda_memcpy_pack_unpack(int);

int main()
{
    const int buf_counts[] = {0, 1, 2, 4, 8, 16, 32, 64, 128, 256,
                              512, 1024, 2048, 4096, 8192, 16384,
                              32768, 65536, 131072, 262144, 524288,
                              1048576, 2097152, 4194304, 8388608,
                              16777216, 33554432};
    int memcpy_times[NUM_SIZES];
    double memcpy_avg;

    srand(time(NULL));

    for(int i = 0; i < NUM_SIZES; i++)
    {
        memcpy_times[i] = cuda_memcpy_pack_unpack(buf_counts[i]);
    }

    for(int i = 0; i < NUM_SIZES; i++)
    {
        printf("%d,%d\n", buf_counts[i], memcpy_times[i]);
    }

    return 0;
}

int cuda_memcpy_pack_unpack(int buf_count)
{
    int runs = 100;
    double total_time = 0;
    int buf_size = buf_count * sizeof(int);
    int *input, *pack_buf, *unpack_buf;

    input = (int*)malloc(buf_size);
    cudaMalloc((void**)&pack_buf, buf_size);
    cudaMalloc((void**)&unpack_buf, buf_size);

    for(int i = 0; i < runs; i++)
    {
        clock_t begin = clock();

        cudaMemcpy(pack_buf, input, buf_size, cudaMemcpyHostToDevice);
        cudaMemcpy(unpack_buf, pack_buf, buf_size, cudaMemcpyDeviceToHost);

        clock_t end = clock();
        total_time += (double)(end - begin) / CLOCKS_PER_SEC * 1000000;
    }

    cudaFree(input);
    cudaFree(pack_buf);
    cudaFree(unpack_buf);

    return (int)total_time / runs;
}
