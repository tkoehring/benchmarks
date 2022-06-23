#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <assert.h>
#include <string.h>

#include "yaksa_install/include/yaksa.h"

#define RUNS 1000
#define CONTIG_TYPE YAKSA_TYPE__INT

double pack_unpack(int);

int main()
{
    int rc;
    int buf_count = 262144;
    int buf_size = buf_count * sizeof(int);
    int *input = (int*)malloc(buf_size); /* initialized with data from previous example */
    int *pack_buf = (int*)malloc(buf_size);
    int *unpack_buf = (int*)malloc(buf_size);
    yaksa_info_t yaksa_info = NULL;
    yaksa_request_t request;
    yaksa_type_t contig = YAKSA_TYPE__INT;
    uintptr_t actual_pack_bytes;
    uintptr_t actual_unpack_bytes;
    double total_time = 0.0;
    double avg_time = 0.0;

    yaksa_init(NULL);
    srand(time(NULL));

    /* Create random buffer */
    for(int i=0; i < buf_count; i++)
    {
        input[i] = rand();    
    }

    printf("Beginning Runs\n");
    for(int i = 0; i < RUNS; i++)
    {
        clock_t begin = clock();

        /* start packing */
        rc = yaksa_ipack(input, buf_count, contig, 0,
                        pack_buf, buf_size, &actual_pack_bytes, yaksa_info,
                        YAKSA_OP__REPLACE, &request);
        assert(rc == YAKSA_SUCCESS);

        /* wait for packing to complete */
        rc = yaksa_request_wait(request);
        assert(rc == YAKSA_SUCCESS);

        /* start unpacking */
        rc = yaksa_iunpack(pack_buf, buf_size, unpack_buf, buf_count, contig, 0,
                           &actual_unpack_bytes, yaksa_info,
                           YAKSA_OP__REPLACE, &request);
        assert(rc == YAKSA_SUCCESS);

        /* wait for unpacking to complete */
        rc = yaksa_request_wait(request);
        assert(rc == YAKSA_SUCCESS);
        clock_t end = clock();
        total_time += (double)(end - begin) / CLOCKS_PER_SEC;
        if(validate_buffer(input, unpack_buf, buf_count) < 1)
            printf("Failed Validation\n");
    }

    printf("Average Yaksa Pack/Unpack Time: %f\n", total_time / RUNS);

    // Memcpy benchmark
    total_time = 0;
    for(int i = 0; i < RUNS; i++)
    {
        
        clock_t begin = clock();

        memcpy(pack_buf, input, buf_size);
        memcpy(unpack_buf, pack_buf, buf_size);

        clock_t end = clock();
        total_time += (double)(end - begin) / CLOCKS_PER_SEC;
    }

    printf("Average Memcpy Pack/Unpack Time: %f\n", total_time / RUNS);


    yaksa_finalize();

    return 0;
}

double pack_unpack(int buf_count)
{
    int rc;
    int buf_size = buf_count * sizeof(int);
    int *input = (int*)malloc(buf_size); /* initialized with data from previous example */
    int *pack_buf = (int*)malloc(buf_size);
    int *unpack_buf = (int*)malloc(buf_size);
    yaksa_info_t yaksa_info = NULL;
    yaksa_request_t request;
    yaksa_type_t contig = YAKSA_TYPE__INT;
    uintptr_t actual_pack_bytes;
    uintptr_t actual_unpack_bytes;
    double total_time = 0.0;
    double avg_time = 0.0;


    /* Create random buffer */
    for(int i=0; i < buf_count; i++)
    {
        input[i] = rand();    
    }

    printf("Beginning Runs\n");
    for(int i = 0; i < RUNS; i++)
    {
        clock_t begin = clock();

        /* start packing */
        rc = yaksa_ipack(input, buf_count, contig, 0,
                        pack_buf, buf_size, &actual_pack_bytes, yaksa_info,
                        YAKSA_OP__REPLACE, &request);
        assert(rc == YAKSA_SUCCESS);

        /* wait for packing to complete */
        rc = yaksa_request_wait(request);
        assert(rc == YAKSA_SUCCESS);

        /* start unpacking */
        rc = yaksa_iunpack(pack_buf, buf_size, unpack_buf, buf_count, contig, 0,
                           &actual_unpack_bytes, yaksa_info,
                           YAKSA_OP__REPLACE, &request);
        assert(rc == YAKSA_SUCCESS);

        /* wait for unpacking to complete */
        rc = yaksa_request_wait(request);
        assert(rc == YAKSA_SUCCESS);
        clock_t end = clock();
        total_time += (double)(end - begin) / CLOCKS_PER_SEC;
        if(validate_buffer(input, unpack_buf, buf_count) < 1)
            printf("Failed Validation\n");
    }
    
    free(input);
    free(pack_buf);
    free(unpack_buf);
}
