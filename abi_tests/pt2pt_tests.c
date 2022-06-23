#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <time.h>
#include <mpi.h>

#define BUF_COUNT 256
int send_recv(int);

int main()
{
    int world_size, rank;
    int errors = 0;

    // Init
    MPI_Init(NULL, NULL);

    // Rank & Comm
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if(world_size != 2)
    {
        if(rank == 0)
        {
            printf("Error two ranks required.\n");
        }

        exit(0);
    }

    errors += send_recv(rank);

    printf("Errors: %d\n", errors);
}

int send_recv(int rank)
{
    int *buf = (int*)malloc(sizeof(int)*BUF_COUNT);
    int err;

    if(rank == 0)
    {
        srand(time(NULL));

        for(int i = 0; i < BUF_COUNT; i++)
        {
            buf[i] = rand();
        }

        err = MPI_Send(buf, 256, MPI_INT, 1, 0, MPI_COMM_WORLD);
    }
    else if(rank == 1)
    {
        err = MPI_Recv(buf, 256, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    return err;
}

