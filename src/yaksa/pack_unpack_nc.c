#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <assert.h>
#include <string.h>

#include "yaksa.h"

//#define NUM_SIZES 27
#define NUM_SIZES 1
#define CONTIG_TYPE YAKSA_TYPE__INT

double pack_unpack_nc(int);
double memcpy_pack_unpack(int);

int main()
{
    /*
    const int buf_counts[] = {0, 1, 2, 4, 8, 16, 32, 64, 128, 256,
                              512, 1024, 2048, 4096, 8192, 16384,
                              32768, 65536, 131072, 262144, 524288,
                              1048576, 2097152, 4194304, 8388608,
                              16777216, 33554432};
    */
    const int buf_counts[] = {64};
    double yaksa_times[NUM_SIZES];
    double memcpy_times[NUM_SIZES];

    double yaksa_avg;
    double memcpy_avg;

    yaksa_init(NULL);
    srand(time(NULL));

    for(int i = 0; i < NUM_SIZES; i++)
    {
        yaksa_times[i] = pack_unpack_nc(buf_counts[i]);
        //memcpy_times[i] = memcpy_pack_unpack(buf_counts[i]);
    }

    for(int i = 0; i < NUM_SIZES; i++)
    {
        printf("%d,%.10f,%.10f\n", buf_counts[i],
                                    yaksa_times[i]);
    }

    yaksa_finalize();

    return 0;
}

double pack_unpack_nc(int buf_count)
{
    int rc;
    int buf_size = buf_count * sizeof(int);
    int *input = (int*)malloc(buf_size);
    int *pack_buf = (int*)malloc(buf_size);
    int *unpack_buf = (int*)malloc(buf_size);
    yaksa_request_t request;
    yaksa_type_t vector;
    uintptr_t actual_pack_bytes;

    for(int i=0; i < buf_count; i++)
    {
        input[i] = rand();    
    }

    /* start packing.
     * note that we can request more bytes in max_pack_bytes and will
     * get the correct number of packed bytes in actual_pack_bytes */
    rc = yaksa_ipack(input, 1, vector, 256, pack_buf,
                     &actual_pack_bytes, &request);
    assert(rc == YAKSA_SUCCESS);

    /* wait for packing to complete */
    rc = yaksa_request_wait(request);
    assert(rc == YAKSA_SUCCESS);

    /* start unpacking */
    rc = yaksa_iunpack(pack_buf, 32, unpack_buf, 1, vector, 0,
                       &request);
    assert(rc == YAKSA_SUCCESS);

    /* wait for unpacking to complete */
    rc = yaksa_request_wait(request);
    assert(rc == YAKSA_SUCCESS);

    yaksa_type_free(vector);
}

double memcpy_pack_unpack(int buf_count)
{
    int runs;
    int buf_size = buf_count * sizeof(int);
    int *input = (int*)malloc(buf_size); /* initialized with data from previous example */
    int *pack_buf = (int*)malloc(buf_size);
    int *unpack_buf = (int*)malloc(buf_size);
    double total_time = 0;

    /* Create random buffer */
    for(int i=0; i < buf_count; i++)
    {
        input[i] = rand();    
    }

    //Do 1000 runs for messages < 1MB and 100 for greater
    if(buf_count > 262144){
        runs = 100;
    }else{
        runs = 1000;
    }

    for(int i = 0; i < runs; i++)
    {
        
        clock_t begin = clock();

        memcpy(pack_buf, input, buf_size);
        memcpy(unpack_buf, pack_buf, buf_size);

        clock_t end = clock();
        total_time += (double)(end - begin) / CLOCKS_PER_SEC;
    }

    free(input);
    free(pack_buf);
    free(unpack_buf);

    return total_time / runs;
}
