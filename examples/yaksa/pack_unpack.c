#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <assert.h>

#include "yaksa_install/include/yaksa.h"

#define BUF_SIZE 64
#define MAX_INT 100
void print_matrix(int*, int, int);

int main()
{
    int rc;
    int input_matrix[BUF_SIZE]; /* initialized with data from previous example */
    int pack_buf[BUF_SIZE];
    int unpack_buf[BUF_SIZE];
    yaksa_type_t contig = YAKSA_TYPE__INT;
    yaksa_info_t yaksa_info;
    yaksa_request_t request;
    uintptr_t actual_pack_bytes;
    uintptr_t actual_unpack_bytes;

    yaksa_init(NULL);
    srand(time(NULL));

    /* Create random matrix */
    for(int i=0; i < BUF_SIZE; i++)
    {
        input_matrix[i] = rand() % MAX_INT;    
    }

    printf("** Input Matrix **\n");
    print_matrix(input_matrix, 8, 8);
    printf("\n");

    /* start packing */
    rc = yaksa_ipack(input_matrix, BUF_SIZE, contig, 0,
                    pack_buf, 256, &actual_pack_bytes, yaksa_info,
                    YAKSA_OP__REPLACE, &request);
    assert(rc == YAKSA_SUCCESS);

    /* wait for packing to complete */
    rc = yaksa_request_wait(request);
    assert(rc == YAKSA_SUCCESS);

    /* start unpacking */
    rc = yaksa_iunpack(pack_buf, 256, unpack_buf, 64, contig, 0,
                       &actual_unpack_bytes, yaksa_info,
                       YAKSA_OP__REPLACE, &request);
    assert(rc == YAKSA_SUCCESS);

    /* wait for unpacking to complete */
    rc = yaksa_request_wait(request);
    assert(rc == YAKSA_SUCCESS);

    yaksa_type_free(contig);

    printf("** Unpacked Matrix **\n");
    print_matrix(unpack_buf, 8, 8);

    yaksa_finalize();
    return 0;
}

void print_matrix(int *mat, int col, int row)
{

    for(int i = 0; i < row; i++)
    {
        for(int j = 0; j < col; j++)
        {
            printf("%d ", mat[j + (col * i)]);
        }
        printf("\n");
    }
}
