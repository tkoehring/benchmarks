#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>

#define msg_size 128

int main(int argc, char **argv)
{
	int rank, size;
	char *buffer;

	MPI_Init(NULL, NULL);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	buffer = (char*)malloc(sizeof(char) * msg_size);

	if(rank == 0)
	{
		for(int i = 1; i < size; i++)
		{
			MPI_Recv(buffer, msg_size, MPI_BYTE,  i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			printf("Recv: %s\n", buffer);
		}

	}
	else
	{
		memcpy(buffer, "Hello", sizeof("Hello")); 
		MPI_Send(buffer, msg_size, MPI_BYTE, 0, 0, MPI_COMM_WORLD);
	}

	MPI_Finalize();
	return 0;
}
