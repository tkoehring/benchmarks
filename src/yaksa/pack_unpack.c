#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>
#include <assert.h>
#include <string.h>

#include <cuda_runtime_api.h>

#include "yaksa.h"

#define RUNS 100
#define NUM_SIZES 27

void parse_arguments(int, char**);
double pack_unpack(int);
void *create_buf(int);
void free_buf(void*);

//Set by -g flag, use a gpu buffer
int _gpu = 0;

int main(int argc, char **argv)
{
    const int buf_counts[] = {0, 1, 2, 4, 8, 16, 32, 64, 128, 256,
                              512, 1024, 2048, 4096, 8192, 16384,
                              32768, 65536, 131072, 262144, 524288,
                              1048576, 2097152, 4194304, 8388608,
                              16777216, 33554432};

    double yaksa_times[NUM_SIZES];
    double yaksa_avg;

    yaksa_init(NULL);
    srand(time(NULL));

    for(int i = 0; i < NUM_SIZES; i++)
    {
        yaksa_times[i] = pack_unpack(buf_counts[i]);
    }

    for(int i = 0; i < NUM_SIZES; i++)
    {
        printf("%d,%.10f\n", buf_counts[i], yaksa_times[i]);
    }

    yaksa_finalize();

    return 0;
}

void parse_arguments(int argc, char **argv){
    int opt;
    while((opt = getopt (argc, argv, "g")) != -1){
        switch(opt){
            case 'g':
                _gpu = 1;             
        }
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

    /* Populate input buffer */
    for(int i=0; i < buf_count; i++)
    {
        input[i] = rand();    
    }
    
    for(int i = 0; i < RUNS; i++)
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

    return total_time / RUNS;
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
    } else {
        free(buf);
    }
}
