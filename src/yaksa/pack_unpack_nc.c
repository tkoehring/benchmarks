#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <assert.h>
#include <string.h>

#include "yaksa.h"

#define MAX_COUNT 33554432
#define BLOCK_LENGTH 1
#define STRIDE 4
#define BUF_COUNT MAX_COUNT * STRIDE
#define BUF_SIZE BUF_COUNT * sizeof(int)
#define DATA_TYPE YAKSA_TYPE__INT
#define NUM_SIZES 27

int pack_unpack_nc(int*, int*, int*, int, int, int);

int main()
{
    int yaksa_times[NUM_SIZES];
    double yaksa_avg;

    int *input = (int*)malloc(BUF_SIZE);
    int *pack_buf = (int*)malloc(BUF_SIZE);
    int *unpack_buf = (int*)malloc(BUF_SIZE);

    const int counts[] = {0, 1, 2, 4, 8, 16, 32, 64, 128, 256,
                              512, 1024, 2048, 4096, 8192, 16384,
                              32768, 65536, 131072, 262144, 524288,
                              1048576, 2097152, 4194304, 8388608,
                              16777216, 33554432};
    yaksa_init(NULL);
    srand(time(NULL));

    for(int i=0; i < BUF_COUNT; i++)
    {
        input[i] = rand();    
    }

    for(int i = 0; i < NUM_SIZES; i++)
    {
        yaksa_times[i] = pack_unpack_nc(input, pack_buf, unpack_buf,
                                        counts[i], 1, 4);
    }

    for(int i = 0; i < NUM_SIZES; i++)
    {
        printf("%d,%d\n", counts[i], yaksa_times[i]);
    }

    yaksa_finalize();
    free(input);
    free(pack_buf);
    free(unpack_buf);
    return 0;
}

int pack_unpack_nc(int *input, int *pack_buf, int *unpack_buf,
                      int count, int block_length, int stride)
{
    int rc;
    yaksa_info_t yaksa_info = NULL;
    yaksa_request_t request;
    yaksa_type_t vector;
    uintptr_t actual_pack_bytes, actual_unpack_bytes;
    double total_time = 0.0;
    int runs = 100;

    rc = yaksa_type_create_vector(count, block_length, stride,
                                  DATA_TYPE, NULL, &vector);
    assert(rc == YAKSA_SUCCESS);

    for(int i = 0; i < runs; i++)
    {
        clock_t begin = clock();

        rc = yaksa_ipack(input, 1, vector, 0, pack_buf,
                         BUF_SIZE, &actual_pack_bytes, yaksa_info,
                         YAKSA_OP__REPLACE, &request);
        assert(rc == YAKSA_SUCCESS);

        rc = yaksa_request_wait(request);
        assert(rc == YAKSA_SUCCESS);

        assert(actual_pack_bytes == count * sizeof(int));

        rc = yaksa_iunpack(pack_buf, actual_pack_bytes, unpack_buf, 1, vector,
                           0, &actual_unpack_bytes, yaksa_info,
                           YAKSA_OP__REPLACE, &request);
        assert(rc == YAKSA_SUCCESS);

        rc = yaksa_request_wait(request);
        assert(rc == YAKSA_SUCCESS);

        clock_t end = clock();
        total_time += (double)(end - begin) / CLOCKS_PER_SEC * 1000000;
    }

    yaksa_type_free(vector);
    return (int)total_time / runs;
}
