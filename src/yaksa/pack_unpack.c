#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <math.h>
#include <time.h>
#include <assert.h>
#include <string.h>
#include <cuda_runtime_api.h>
#include "yaksa.h"

void parse_arguments(int, char**);
void warmup();
void pack_unpack(int, int*, int*);
void *create_buf(int, int);
void free_buf(void*, int);

//Set by the -i flag. H for host buffer, G for device buffer. Host as default
int _input_gpu = 0;

//Set by the -o flag. H for host buffer, G for device buffer. Host as default
int _output_gpu = 0;

//Set by -r <val> flag. Indicates how many runs to perform
int _runs = 0;

//Set by the -M <val> flag. Indicates how big the buffers should be. Argument
//asks for size in megabytes, then converted to bytes
int _size = 0;

int main(int argc, char **argv)
{
    parse_arguments(argc, argv);

    int num_sizes = (log(_size) / log(2) + 2);
    int *buf_counts = (int*)malloc(sizeof(int) * num_sizes);
    int *yaksa_pack_times = (int*)malloc(sizeof(int) * num_sizes);
    int *yaksa_unpack_times = (int*)malloc(sizeof(int) * num_sizes);
    double yaksa_avg;

    buf_counts[0] = 0;
    buf_counts[1] = 1;

    for(int i = 2; i < num_sizes; i++){
        buf_counts[i] = (2 << (i - 2));
    }

    yaksa_init(NULL);
    srand(time(NULL));

    warmup();

    for(int i = 0; i < num_sizes; i++)
    {
        pack_unpack(buf_counts[i],
                    &(yaksa_pack_times[i]),
                    &(yaksa_unpack_times[i]));
    }

    printf("Memory (Bytes), yaksa_ipack (us), yaksa_iunpack (us)\n");

    for(int i = 0; i < num_sizes; i++)
    {
        printf("%d, %d, %d\n", buf_counts[i],
                          yaksa_pack_times[i],
                          yaksa_unpack_times[i]);
    }

    free(buf_counts);
    free(yaksa_pack_times);
    free(yaksa_unpack_times);
    yaksa_finalize();

    return 0;
}

void parse_arguments(int argc, char **argv){
    int opt;
    while((opt = getopt (argc, argv, "i:M:o:r:")) != -1){
        switch(opt){
            case 'i':
                if(*optarg == 'D'){
                    _input_gpu = 1;
                }
                break;
            case 'M':
                // Multiply the number of bytes in 1 MB
                _size = atoi(optarg) * 1048576;
                break;
            case 'o':
                if(*optarg == 'D'){
                    _output_gpu = 1;
                }
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

// Warmup loop of 100 runs for a 1 byte message size
void warmup(){
    int rc;
    yaksa_info_t yaksa_info = NULL;
    yaksa_request_t request;
    yaksa_type_t contig = YAKSA_TYPE__INT;
    uintptr_t actual_pack_bytes;
    uintptr_t actual_unpack_bytes;
    int buf_count = 1;
    int buf_size = buf_count * sizeof(int);
    int *input = (int*)create_buf(buf_size, _input_gpu);
    int *pack_buf = (int*)create_buf(buf_size, _output_gpu);
    int *unpack_buf = (int*)create_buf(buf_size, _output_gpu);

    for(int i = 0; i < 100; i++)
    {
        //Start packing
        rc = yaksa_ipack(input, buf_count, YAKSA_TYPE__INT, 0,
                        pack_buf, buf_size, &actual_pack_bytes, yaksa_info,
                        YAKSA_OP__REPLACE, &request);
        assert(rc == YAKSA_SUCCESS);

        //Wait for packing to complete
        rc = yaksa_request_wait(request);
        assert(rc == YAKSA_SUCCESS);
            
        //Start unpacking
        rc = yaksa_iunpack(input, buf_size, unpack_buf, buf_count,
                           YAKSA_TYPE__INT, 0, &actual_unpack_bytes,
                           yaksa_info, YAKSA_OP__REPLACE, &request);
        assert(rc == YAKSA_SUCCESS);

        //Wait for unpacking to complete
        rc = yaksa_request_wait(request);
        assert(rc == YAKSA_SUCCESS);
    }
}

void pack_unpack(int buf_count, int *pack_time, int *unpack_time)
{
    int rc;
    yaksa_info_t yaksa_info = NULL;
    yaksa_request_t request;
    yaksa_type_t contig = YAKSA_TYPE__INT;
    uintptr_t actual_pack_bytes;
    uintptr_t actual_unpack_bytes;

    int buf_size = buf_count * sizeof(int);
    int *input = (int*)create_buf(buf_size, _input_gpu);
    int *pack_buf = (int*)create_buf(buf_size, _output_gpu);
    int *unpack_buf = (int*)create_buf(buf_size, _output_gpu);

    double pack_total = 0.0f, unpack_total = 0.0f;
    clock_t begin, end;

    for(int i = 0; i < _runs; i++)
    {
        //Start packing
        begin = clock();
        rc = yaksa_ipack(input, buf_count, YAKSA_TYPE__INT, 0,
                        pack_buf, buf_size, &actual_pack_bytes, yaksa_info,
                        YAKSA_OP__REPLACE, &request);
        assert(rc == YAKSA_SUCCESS);

        //Wait for packing to complete
        rc = yaksa_request_wait(request);
        assert(rc == YAKSA_SUCCESS);
        end = clock();
        pack_total += (double)(end - begin) / CLOCKS_PER_SEC * 1000000;
            
        //Start unpacking
        begin = clock();
        rc = yaksa_iunpack(input, buf_size, unpack_buf, buf_count,
                           YAKSA_TYPE__INT, 0, &actual_unpack_bytes,
                           yaksa_info, YAKSA_OP__REPLACE, &request);
        assert(rc == YAKSA_SUCCESS);

        //Wait for unpacking to complete
        rc = yaksa_request_wait(request);
        assert(rc == YAKSA_SUCCESS);
        end = clock();
        unpack_total += (double)(end - begin) / CLOCKS_PER_SEC * 1000000;
    }

    *pack_time = pack_total / _runs;
    *unpack_time = unpack_total / _runs;

    free_buf(input, _input_gpu);
    free_buf(pack_buf, _output_gpu);
    free_buf(unpack_buf, _output_gpu);
}

void *create_buf(int buf_size, int gpu){
    if(gpu){
        void *buf;
        (void*)cudaMalloc((void**)&buf, buf_size);
        return buf;
    } else {
        return (void*)malloc(buf_size);
    }
}

void free_buf(void *buf, int gpu){
    if(gpu){
        cudaFree(buf);
        return;
    } else {
        free(buf);
        return;
    }
}
