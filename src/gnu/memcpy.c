#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <assert.h>
#include <string.h>

#define NUM_SIZES 27

double memcpy_pack_unpack(int);

int main()
{
    const int buf_counts[] = {0, 1, 2, 4, 8, 16, 32, 64, 128, 256,
                              512, 1024, 2048, 4096, 8192, 16384,
                              32768, 65536, 131072, 262144, 524288,
                              1048576, 2097152, 4194304, 8388608,
                              16777216, 33554432};
    double memcpy_times[NUM_SIZES];
    double memcpy_avg;

    srand(time(NULL));

    for(int i = 0; i < NUM_SIZES; i++)
    {
        memcpy_times[i] = memcpy_pack_unpack(buf_counts[i]);
    }

    for(int i = 0; i < NUM_SIZES; i++)
    {
        printf("%d,%.10f\n", buf_counts[i], memcpy_times[i]);
    }

    return 0;
}

double memcpy_pack_unpack(int buf_count)
{
    int runs;
    int buf_size = buf_count * sizeof(int);
    int *input = (int*)malloc(buf_size);
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
