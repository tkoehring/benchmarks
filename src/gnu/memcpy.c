#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <math.h>
#include <time.h>
#include <assert.h>
#include <string.h>

//Set by -r <val> flag. Indicates how many runs to perform
int _runs = 0;

//Set by the -M <val> flag. Indicates how big the buffers should be. Argument
//asks for size in megabytes, then converted to bytes
int _size = 0;

void parse_arguments(int, char**);
double memcpy_benchmark(int);

int main(int argc, char **argv)
{
    parse_arguments(argc, argv);

    int num_sizes = (log(_size) / log(2) + 2);
    int *buf_counts = (int*)malloc(sizeof(int) * num_sizes);
    double *memcpy_times = (double*)malloc(sizeof(double) * num_sizes);
    double memcpy_avg;
    
    srand(time(NULL));

    buf_counts[0] = 0;
    buf_counts[1] = 1;

    for(int i = 2; i < num_sizes; i++){
        buf_counts[i] = (2 << (i - 2));
    }

    for(int i = 0; i < num_sizes; i++)
    {
        memcpy_times[i] = memcpy_benchmark(buf_counts[i]);
    }

    printf("Memory (Bytes), memcpy (us)\n");

    for(int i = 0; i < num_sizes; i++)
    {
        printf("%d, %.3f\n", buf_counts[i], memcpy_times[i]);
    }

    free(buf_counts);
    free(memcpy_times);
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

double memcpy_benchmark(int buf_count)
{
    int buf_size = buf_count * sizeof(int);
    int *input = (int*)malloc(buf_size);
    int *output = (int*)malloc(buf_size);
    double total_time = 0;

    for(int i = 0; i < _runs; i++)
    {
        clock_t begin = clock();

        memcpy(output, input, buf_size);

        clock_t end = clock();
        total_time += (double)(end - begin) / CLOCKS_PER_SEC * 1000000;
    }

    free(input);
    free(output);

    return total_time / _runs;
}
