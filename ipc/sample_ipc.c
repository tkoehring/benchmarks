#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <cuda_runtime_api.h>
#include <string.h>

#define count 8192 //The minimum number of ints at a byte size of 4 to trigger an IPC pathway

int main(int argc, char **argv)
{
	int rank;
	int *dev_buf;

	MPI_Init(NULL, NULL);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	cudaMalloc((void**) &dev_buf, sizeof(int) * count * 2 * 2);


	if(rank == 0)
	{
		int buf[count];

		//Recv as int instead of the vector data type.
		MPI_Recv(dev_buf, count, MPI_INT, 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

		//Copy from the gpu buffer to local buffer
		cudaMemcpy(buf, dev_buf, sizeof(int) * count, cudaMemcpyDeviceToHost);

		printf("Receiving\n");
	}
	else
	{
		// Create a count by two matrix. We will create a NC vector with a stride of two out of the matrix.
		int data[count * 2][2];
		
		for(int i = 0; i < count * 2; i++)
		{
			for(int j = 0; j < 2; j++)
			{
				data[i][j] = rand() % 100;
			}
		}

		//Create strided data type
		MPI_Datatype new_type;
		MPI_Type_vector(count, 1, 2, MPI_INT, &new_type);
		MPI_Type_commit(&new_type);

		//Copy matrix to gpu buffer
		cudaMemcpy(dev_buf, data, sizeof(int) * count * 2 * 2, cudaMemcpyHostToDevice);

		//Send only one of the new data type
		MPI_Send(dev_buf, 1, new_type, 0, 0, MPI_COMM_WORLD);

		printf("Sending\n");
	}

	MPI_Finalize();

	return 0;
}
