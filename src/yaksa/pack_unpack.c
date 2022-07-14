#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>
#include <assert.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime_api.h>
#include "yaksa.h"

#define NUM_SIZES 27

void parse_arguments(int, char**);
double pack_unpack(int);
void *create_buf(int);
void free_buf(void*);

//Set by -g flag. Use a gpu buffer
int _gpu = 0;

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
    double *yaksa_times = (double*)malloc(sizeof(double) * num_sizes);
    double yaksa_avg;

    buf_counts[0] = 0;
    buf_counts[1] = 1;

    for(int i = 2; i < num_sizes; i++){
        buf_counts[i] = (2 << (i - 2));
    }


    yaksa_init(NULL);
    srand(time(NULL));

    for(int i = 0; i < num_sizes; i++)
    {
        yaksa_times[i] = pack_unpack(buf_counts[i]);
    }

    for(int i = 0; i < num_sizes; i++)
    {
        printf("%d,%.10f\n", buf_counts[i], yaksa_times[i]);
    }

    free(buf_counts);
    free(yaksa_times);
    yaksa_finalize();

    return 0;
}

void parse_arguments(int argc, char **argv){
    int opt;
    while((opt = getopt (argc, argv, "gM:r:")) != -1){
        switch(opt){
            case 'g':
                _gpu = 1;
                break;
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

double pack_unpack(int buf_count)
{
    int rc;
    yaksa_info_t yaksa_info = NULL;
    yaksa_request_t request;
    yaksa_type_t contig = YAKSA_TYPE__INT;
    uintptr_t actual_pack_bytes;
    uintptr_t actual_unpack_bytes;

    int buf_size = buf_count * sizeof(int);

    int *input = (int*)create_buf(buf_size);
    int *pack_buf = (int*)create_buf(buf_size);
    int *unpack_buf = (int*)create_buf(buf_size);

    double total_time = 0.0;

    for(int i = 0; i < _runs; i++)
    {
        //Start timer
        clock_t begin = clock();

        //Start packing
        rc = yaksa_ipack(input, buf_count, YAKSA_TYPE__INT, 0,
                        pack_buf, buf_size, &actual_pack_bytes, yaksa_info,
                        YAKSA_OP__REPLACE, &request);
        assert(rc == YAKSA_SUCCESS);

        //Wait for packing to complete
        rc = yaksa_request_wait(request);
        assert(rc == YAKSA_SUCCESS);
        //Start unpacking
        rc = yaksa_iunpack(pack_buf, buf_size, unpack_buf, buf_count,
                           YAKSA_TYPE__INT, 0, &actual_unpack_bytes,
                           yaksa_info, YAKSA_OP__REPLACE, &request);
        assert(rc == YAKSA_SUCCESS);

        //Wait for unpacking to complete
        rc = yaksa_request_wait(request);
        assert(rc == YAKSA_SUCCESS);
        //Stop timer
        clock_t end = clock();
        total_time += (double)(end - begin) / CLOCKS_PER_SEC;
    }

    free_buf(input);
    free_buf(pack_buf);
    free_buf(unpack_buf);

    return total_time / _runs;
}

void *create_buf(int buf_size){
    if(_gpu){
        void *buf;
        (void*)cudaMalloc((void**)&buf, buf_size);
        return buf;
    } else {
        return (void*)malloc(buf_size);
    }
}

void free_buf(void *buf){
    if(_gpu){
        cudaFree(buf);
        return;
    } else {
        free(buf);
        return;
    }
}
