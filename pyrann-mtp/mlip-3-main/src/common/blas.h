/*   MLIP is a software for Machine Learning Interatomic Potentials
 *   MLIP is released under the "New BSD License", see the LICENSE file.
 *   Contributors: Alexander Shapeev, Evgeny Podryabinkin
 */
#ifndef MLIP_BLAS
#define MLIP_BLAS


#if defined(_WIN32) || defined(MLIP_NOBLAS) 
#   include <cmath>
#   include <algorithm>

// v1[i] += s * v2[i]. v1, v2 - vectors of length 'size', s - scalar
inline void BLAS_v1_add_s_mult_v2(double* v1, double s, double* v2, int size, 
                                    int inc1=1, int inc2=1)
{
    for (int i=0; i<size; i++)
        v1[i*inc1] += s * v2[i*inc2];
}

// returns index of maximal in modulus element in vector 'v'
inline int BLAS_find_max(double* v, int size)
{
    int ind_max=0;
    double max = -1.0;
    for (int i=0; i<size; i++)
        if (fabs(v[i]) > max)
        {
            max = fabs(v[i]);
            ind_max = i;
        }
    return ind_max;
}

// swaps two arrays
inline void BLAS_swap_arrays(double* v1, double* v2, int size)
{
    for (int i=0; i<size; i++)
    {
        double tmp = v1[i];
        v1[i] = v2[i];
        v2[i] = tmp;
    }
}

// v1[i] = s*M[i][j]*v2[j]. s - scalar, v1,v2 - vectors, M - matrix of size 'row_count' x 'col_count'
inline void BLAS_v1_equal_s_mult_M_mult_v2(double* v1, double s, double* M, double* v2, 
                         int row_count, int col_count)
{
    for (int i=0; i<row_count; i++)
    {
        v1[i] = 0.0;
        for (int j=0; j<col_count; j++)
            v1[i] += s * M[i*col_count+j] * v2[j];
    }
}

// M[i][j] += s * v1[i] * v2[j]^T. Rank-1 update, M - matrix row_count x col_count, s - scalar, v1,v2 - vectors
inline void BLAS_M_add_s_mult_v1_mult_v2(double* M, double s, double* v1, double* v2, 
                                    int row_count, int col_count)
{
    for (int i=0; i<row_count; i++)
        for (int j=0; j<col_count; j++)
            M[i*col_count+j] += s * v1[i] * v2[j];
}

// M3[i][j] = M1[i][k] * M2[j][k]^T. M1 is row_count*lost_size, M2 is col_count*lost_size, M3 is row_count*col_count. All matrices are written by rows
inline void BLAS_M3_equal_M1_mult_M2T(double* M1, double* M2, double* M3, 
                                int row_count, int col_count, int lost_size)
{   
    for (int i=0; i<row_count; i++)
        for (int j=0; j<col_count; j++)
        {
            M3[i*col_count+j] = 0.0;
            for (int l=0; l<lost_size; l++)
                M3[i*col_count+j] += M1[i*lost_size+l] * M2[j*lost_size+l];
        }
}

#else
#   ifdef MLIP_INTEL_MKL
#       include <mkl_cblas.h>
#   else 
#       include <cblas.h>
#   endif
#   include <memory.h>

 // v1[i] += s * v2[i]. v1, v2 - vectors of length 'size', s - scalar
inline void BLAS_v1_add_s_mult_v2(double* v1, double s, double* v2, int size,
                                    int inc1=1, int inc2=1)
{
    cblas_daxpy(size, s, v2, inc2, v1, inc1);
}

// returns index of maximal in modulus element in vector 'vec'
inline int BLAS_find_max(double* v, int size)
{
    return (int)cblas_idamax(size, v, 1);
}

// swaps two arrays
inline void BLAS_swap_arrays(double* v1, double* v2, int size)
{
    cblas_dswap(size, v1, 1, v2, 1);
}

// v1[i] = s*M[i][j]*v2[j]. s - scalar, v1,v2 - vectors, M - matrix of size 'row_count' x 'col_count'
inline void BLAS_v1_equal_s_mult_M_mult_v2(double* v1, double s, double* M, double* v2, 
                                            int row_count, int col_count)
{
    memset(v1, 0, row_count*sizeof(double));
    cblas_dgemv(CBLAS_ORDER::CblasRowMajor, CBLAS_TRANSPOSE::CblasNoTrans,
                row_count, col_count, s, M, col_count, v2, 1, 0.0, v1, 1);
}

// M[i][j] += s * v1[i] * v2^T[j]. Rank-1 update, M - matrix row_count x col_count, s - scalar, v1,v2 - vectors
inline void BLAS_M_add_s_mult_v1_mult_v2(double* M, double s, double* v1, double* v2,
                                         int row_count, int col_count)
{
    cblas_dger(CBLAS_ORDER::CblasRowMajor, row_count, col_count,
                s, v1, 1, v2, 1, M, col_count);
}

// M3[i][j] = M1[i][k] * M2[j][k]^T. M1 is row_count*lost_size, M2 is col_count*lost_size, M3 is row_count*col_count. All matrices are written by rows
inline void BLAS_M3_equal_M1_mult_M2T(double* M1, double* M2, double* M3, 
                                        int row_count, int col_count, int lost_size)
{
    memset(M3, 0, row_count*col_count*sizeof(double));
    cblas_dgemm(CBLAS_ORDER::CblasRowMajor, CBLAS_TRANSPOSE::CblasNoTrans, CBLAS_TRANSPOSE::CblasTrans,
                row_count, col_count, lost_size, 1.0, M1, lost_size, M2, lost_size, 0.0, M3, col_count);
}

#endif
                   
#endif //MLIP_BLAS
