#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <math.h>
#include <time.h>
#include <assert.h>
#include <string.h>
#include <cuda_runtime_api.h>

//Set by -r <val> flag. Indicates how many runs to perform
int _runs = 0;

//Set by the -M <val> flag. Indicates how big the buffers should be. Argument
//asks for size in megabytes, then converted to bytes
int _size = 0;

void parse_arguments(int, char**);
void cuda_memcpy_benchmark(int, int*, int*, int*);

int main(int argc, char **argv)
{
    parse_arguments(argc, argv);

    int num_sizes = (log(_size) / log(2) + 2);
    int *buf_counts = (int*)malloc(sizeof(int) * num_sizes);
    int *hd_times = (int*)malloc(sizeof(int) * num_sizes);
    int *dd_times = (int*)malloc(sizeof(int) * num_sizes);
    int *dh_times = (int*)malloc(sizeof(int) * num_sizes);
    double memcpy_avg;

    srand(time(NULL));

    buf_counts[0] = 0;
    buf_counts[1] = 1;

    for(int i = 2; i < num_sizes; i++){
        buf_counts[i] = (2 << (i - 2));
    }

    for(int i = 0; i < num_sizes; i++)
    {
        cuda_memcpy_benchmark(buf_counts[i],
                              &(hd_times[i]),
                              &(dd_times[i]),
                              &(dh_times[i]));
    }

    printf("Memory (Bytes), HostToDevice (us), DeviceToHost (us), DeviceToDevice(us)\n");

    for(int i = 0; i < num_sizes; i++)
    {
        printf("%d, %d, %d, %d\n", buf_counts[i],
                                   hd_times[i],
                                   dh_times[i],
                                   dd_times[i]);
    }

    free(buf_counts);
    free(hd_times);
    free(dd_times);
    free(dh_times);

    return 0;
}

void parse_arguments(int argc, char **argv){
    int opt;
    while((opt = getopt (argc, argv, "M:r:")) != -1){
        switch(opt){
            case 'M':
                // Multiply the number of bytes in 1 MB
                _size = atoi(optarg) * 1048576;
                break;
            case'r':
                _runs = atoi(optarg);
                break;
        }
    }

    //Set default value to 1MB
    if(_size <= 0){
        _size = 1048576;
    }

    //Default case for _runs
    if(_runs <= 0){
       _runs = 100;
    }
}

void cuda_memcpy_benchmark(int buf_count,
                          int *hd_time, int *dd_time, int *dh_time)
{
    double hd_total = 0.0f, dd_total = 0.0f, dh_total = 0.0f;
    int buf_size = buf_count * sizeof(int);
    int *input, *pack_buf, *unpack_buf;
    clock_t begin, end;

    input = (int*)malloc(buf_size);
    cudaMalloc((void**)&pack_buf, buf_size);
    cudaMalloc((void**)&unpack_buf, buf_size);

    for(int i = 0; i < _runs; i++)
    {
            begin = clock();
            cudaMemcpy(pack_buf, input, buf_size, cudaMemcpyHostToDevice);
            end = clock();
            hd_total += (double)(end - begin) / CLOCKS_PER_SEC * 1000000;

            begin = clock();
            cudaMemcpy(unpack_buf, pack_buf, buf_size, cudaMemcpyDeviceToDevice);
            end = clock();
            dd_total += (double)(end - begin) / CLOCKS_PER_SEC * 1000000;

            begin = clock();
            cudaMemcpy(input, unpack_buf, buf_size, cudaMemcpyDeviceToHost);
            end = clock();
            dh_total += (double)(end - begin) / CLOCKS_PER_SEC * 1000000;
    }

    *hd_time = hd_total / _runs;
    *dd_time = dd_total / _runs;
    *dh_time = dh_total / _runs;

    cudaFree(input);
    cudaFree(pack_buf);
    cudaFree(unpack_buf);
}
