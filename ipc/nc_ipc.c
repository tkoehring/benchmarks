#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <cuda_runtime_api.h>

#define buf_sz 32768

int main(int argc, char **argv)
{
	int rank;
	int *dev_buf;
	int buf[buf_sz];

	MPI_Init(NULL, NULL);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	cudaMalloc((void**) &dev_buf, sizeof(int) * buf_sz);

	if(rank == 0)
	{
		MPI_Recv(dev_buf, buf_sz, MPI_INT, 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		cudaMemcpy(buf, dev_buf, sizeof(int) * buf_sz, cudaMemcpyDeviceToHost);
		printf("Recv: %d\n", buf[0]);
	}
	else
	{
		for(int i = 0; i < buf_sz; i++)
		{
			buf[i] = rand() % 100;
		}

		cudaMemcpy(dev_buf, buf, sizeof(int) * buf_sz, cudaMemcpyHostToDevice);
		printf("Send: %d\n", buf[0]);
		MPI_Send(dev_buf, buf_sz, MPI_INT, 0, 0, MPI_COMM_WORLD);
	}

	MPI_Finalize();

	return 0;
}
