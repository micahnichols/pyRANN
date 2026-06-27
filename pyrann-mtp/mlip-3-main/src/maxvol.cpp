/*   This software is called MLIP for Machine Learning Interatomic Potentials.
 *   MLIP can only be used for non-commercial research and cannot be re-distributed.
 *   The use of MLIP must be acknowledged by citing approriate references.
 *   See the LICENSE file for details.
 *
 *   Contributors: Evgeny Podryabinkin
 */

#include "maxvol.h"
#include "common/utils.h"
#include "common/blas.h"
#include <sstream>
#include <numeric>


using namespace std;


const char* MaxVol::tagname = {"MaxVol"};


void MaxVol::Reset(int _n, double _init_treshold) 
{
    n = _n;
    init_treshold = _init_treshold;

    A.resize(n, n);
    A.set(0);
    for (int i = 0; i < n; i++)
        A(i, i) = init_treshold;

    invA.resize(n, n);
    invA.set(0);
    for (int i=0; i<n; i++)
        invA(i, i) = 1.0/init_treshold;

    B.resize(0, n);
    BinvA.resize(1, n);

    bufVect1.resize(n);
    bufVect2.resize(n);
    bufVect3.resize(n);
    //bufMtrx.resize(n, n);
    
    log_n_det = n*log(init_treshold)/log(n);

    swap_grade = 0;
    i0 = -1;
    j0 = -1;

#ifdef MLIP_MPI
    mpi_swap_root = -1;
#endif
}

MaxVol::MaxVol()
{
    n = 0;
    A.resize(n, n);
    invA.resize(n, n);
    B.resize(0, n);
    BinvA.resize(0, n);
    bufVect1.resize(n);
    bufVect2.resize(n);
    bufVect3.resize(n);
    log_n_det = 0;

    swap_grade = 0;
    i0 = -1;
    j0 = -1;

#ifdef MLIP_MPI
    mpi_swap_root = -1;
#endif
}

MaxVol::MaxVol(int _n, double _init_treshold)
{
    Reset(_n, _init_treshold);
}

MaxVol::~MaxVol()
{
}

void MaxVol::FindMaxElem(Array2D& Mtrx_, int start_row, int row_count)
{
    if (row_count == 0)
        row_count = Mtrx_.size1 - start_row;
#ifdef MLIP_DEBUG
    if (start_row + row_count > Mtrx_.size1)
    {
        cerr << start_row << " " << row_count << " " << Mtrx_.size1 << endl;
        ERROR("Inconsistent arguments");
    }
#endif // MLIP_DEBUG
    if (Mtrx_.size1 > 0)    // regular case
    {
        int ind = BLAS_find_max(&Mtrx_(start_row, 0), row_count*Mtrx_.size2);
        i0 = start_row + ind / Mtrx_.size2;
        j0 = ind % Mtrx_.size2;
        swap_grade = fabs(Mtrx_(i0, j0));
    }
    else        
    {
        i0 = -1;
        j0 = -1;
        swap_grade = 0;
    }
}

#ifdef MLIP_MPI
void MaxVol::MPI_ShareSwapData()
{
    // sharing global maximum and the correspondent row index
    vector<double> swap_grades(mpi.size);
    MPI_Allgather(&swap_grade, 1, MPI_DOUBLE, swap_grades.data(), 1, MPI_DOUBLE, mpi.comm);

    mpi_swap_root = BLAS_find_max(swap_grades.data(), (int)swap_grades.size());
    swap_grade = swap_grades[mpi_swap_root];

    MPI_Bcast(&i0, 1, MPI_INT, mpi_swap_root, mpi.comm);
    MPI_Bcast(&j0, 1, MPI_INT, mpi_swap_root, mpi.comm);
}
#endif // MLIP_MPI

void MaxVol::UpdateInvA()
{
#ifdef MLIP_MPI
    double tmp;

    if (mpi.rank == mpi_swap_root)
    {
        // 0. bufVect1 = B[i0][:] - A[j0][:]
        memcpy(&bufVect1[0], &B(i0, 0), n*sizeof(double));
        BLAS_v1_add_s_mult_v2(&bufVect1[0], -1, &A(j0, 0), n);

        // 1. tmp = (1 + v^T * invA * q) = 1.0/BinvA[i0][j0]
        tmp = 1.0/BinvA(i0, j0);
    }

    MPI_Bcast(bufVect1.data(), n, MPI_DOUBLE, mpi_swap_root, mpi.comm);
    MPI_Bcast(&tmp, 1, MPI_DOUBLE, mpi_swap_root, mpi.comm);
#else
    // 0. bufVect1 = B[i0][:] - A[j0][:]
    memcpy(&bufVect1[0], &B(i0, 0), n*sizeof(double));
    BLAS_v1_add_s_mult_v2(&bufVect1[0], -1, &A(j0, 0), n);

    // SMW - update:
    // 1. tmp = (1 + v^T * invA * q) = 1.0/BinvA[i0][j0]
    double tmp = 1.0/BinvA(i0, j0);
#endif
    // 2. &invA[j0][0] - j0-th column of A^-1
    memcpy(&bufVect2[0], &invA(j0, 0), n*sizeof(double));
    // 3. tmp * bufVect1^T * invA
    BLAS_v1_equal_s_mult_M_mult_v2(&bufVect3[0], tmp, &invA(0, 0), &bufVect1[0], n, n);
    // 4. update invA
    BLAS_M_add_s_mult_v1_mult_v2(&invA(0, 0), -1.0, &bufVect3[0], &bufVect2[0], n, n);
    //cblas_dger(CBLAS_ORDER::CblasColMajor, n, n, -1.0,
    //  &bufVect2[0], 1, &bufVect3[0], 1, &invA(0, 0), n);
}

void MaxVol::UpdateBinvA()
{
#ifdef MLIP_MPI
    int m=BinvA.size1;
    double b_i0j0;
    if (mpi.rank == mpi_swap_root)
    {
        b_i0j0 = BinvA(i0, j0);
        memcpy(&bufVect2[0], &BinvA(i0, 0), n*sizeof(double));
        bufVect2[j0] -= 1.0;
    }

    MPI_Bcast(&b_i0j0, 1, MPI_DOUBLE, mpi_swap_root, mpi.comm);
    MPI_Bcast(bufVect2.data(), n, MPI_DOUBLE, mpi_swap_root, mpi.comm);

    if (m == 0)
        return;
    bufVect1.resize(__max(n, m));
    FillWithZero(bufVect1);
    BLAS_v1_add_s_mult_v2(&bufVect1[0], 1.0/b_i0j0, &BinvA(0, j0), m, 1, n);
    if (mpi.rank == mpi_swap_root)
        bufVect1[i0] += 1.0/b_i0j0;

    BLAS_M_add_s_mult_v1_mult_v2(&BinvA(0, 0), -1.0, &bufVect1[0], &bufVect2[0], m, n);
#else
    // 0. remember BinvA[i0][j0];
    int m=BinvA.size1;
    double b_i0j0 = BinvA(i0, j0);
    // 1. vector-column formation
    bufVect1.resize(__max(n, m));   //not actual resize if m<n
    FillWithZero(bufVect1);
    BLAS_v1_add_s_mult_v2(&bufVect1[0], 1.0/b_i0j0, &BinvA(0, j0), m, 1, n);
    bufVect1[i0] += 1.0/b_i0j0;
    // 2. vector-row formation
    memcpy(&bufVect2[0], &BinvA(i0, 0), n*sizeof(double));
    bufVect2[j0] -= 1.0;
    // 3. BinvA update
    BLAS_M_add_s_mult_v1_mult_v2(&BinvA(0, 0), -1.0, &bufVect1[0], &bufVect2[0], m, n);
#endif
}

void MaxVol::SwapRows()
{
#ifdef MLIP_MPI
    if (mpi.rank == mpi_swap_root)
        BLAS_swap_arrays(&A(j0, 0), &B(i0, 0), n);
    MPI_Bcast(&A(j0, 0), n, MPI_DOUBLE, mpi_swap_root, mpi.comm);
#else
    BLAS_swap_arrays(&A(j0, 0), &B(i0, 0), n);
#endif
}

//void MaxVol::RecalculateInvA()
//{
//  int report;
//  int nxn = n*n;
//  std::vector<int> ipiv(n);
//
//  cblas_domatcopy(CBLAS_ORDER::CblasRowMajor, CBLAS_TRANSPOSE::CblasTrans, 
//      n, n, 1, &A(0,0), n, &invA(0,0), n);
//  LAPACK_dgetrf(&n, &n, &invA(0, 0), &n, ipiv.data(), &report);
//  LAPACK_dgetri(&n, &invA(0, 0), &n, ipiv.data(), &bufMtrx(0, 0), &nxn, &report);
//
//  if (report != 0)
//      ERROR("Maxvol: failed to inverse matrix");
//}

void MaxVol::Iterate()
{
    UpdateInvA();
    UpdateBinvA();
    SwapRows();
#ifdef MLIP_MPI
    swap_tracker.emplace_back(i0, j0, mpi_swap_root);
#else
    swap_tracker.emplace_back(i0, j0);
#endif
    FindMaxElem(BinvA);
#ifdef MLIP_MPI
    MPI_ShareSwapData();
#endif // MLIP_MPI
}

int MaxVol::InitedRowsCount()
{
    if (init_treshold < 0)  // init_treshold = -1 if all rows inited
        return n;

    int cntr = 0;
    for (int i=0; i<n; i++)
        if (A(i, i) != init_treshold)   // only diagonal elements checked. It is not honest but saves performance
            cntr++;

    if (cntr == n)
        init_treshold = -1; // It means that all rows inited

    return cntr;
}

double MaxVol::CalcSwapGrade(std::vector<int>* rows_ranges, std::vector<double>* out_grades)
{
    int m = B.size1;

    BinvA.resize(m, n);
 
    swap_tracker.clear();

    if (m > 0)
        BLAS_M3_equal_M1_mult_M2T(&B(0, 0), &invA(0, 0), &BinvA(0, 0), m, n, n);

    if (rows_ranges == nullptr)
        FindMaxElem(BinvA, 0, m);
    else
    {
#ifdef MLIP_DEBUG   // check if a sum of elements in rows_ranges equal the number of rows in B
        if (out_grades == nullptr)
            ERROR("Inconsistent arguments");
        int sum = accumulate(rows_ranges->begin(), rows_ranges->end(), 0);
        if (sum != B.size1)
            ERROR("Inconsistent argument (rows_ranges)");
#endif // MLIP_DEBUG

        out_grades->resize(rows_ranges->size());

        int summ=0, io=0, jo=0;
        double max_grade = 0.0;
        for (int i=0; i<rows_ranges->size(); i++)
        {
            FindMaxElem(BinvA, summ, rows_ranges->at(i));
            out_grades->at(i) = swap_grade;
            summ += rows_ranges->at(i);
            if (swap_grade > max_grade)
            {
                max_grade = swap_grade;
                io = i0;
                jo = j0;
            }
        }
        swap_grade = max_grade;
        i0 = io;
        j0 = jo;
    }

    std::stringstream logstrm1;
    logstrm1 << "\rMaxVol: call# " << ++eval_cntr
        << '\t' << m << 'x' << n
        << " matrix. First swap grade " << swap_grade << endl;
    MLP_LOG(tagname, logstrm1.str());

    return swap_grade;
}

int MaxVol::MaximizeVol(double threshold, int swap_limit)
{
    if (i0==-1 || j0==-1)
        CalcSwapGrade();

#ifdef MLIP_MPI
    MPI_ShareSwapData();
#endif // MLIP_MPI

    std::stringstream logstrm1;
    for (int swap_cntr=0; swap_grade > threshold && swap_cntr < swap_limit; swap_cntr++)
    {
        log_n_det += log(swap_grade)/log(n);

        Iterate();

        logstrm1 << "\rMaxVol: " << swap_cntr+1 << " swaps. Inited "
            << InitedRowsCount() << "/" << n << ". log(det): "
            << log_n_det << "\t";
        MLP_LOG(tagname, logstrm1.str()); logstrm1.str("");
    }

    if (!swap_tracker.empty())
        logstrm1 << endl;
    MLP_LOG(tagname, logstrm1.str());

    i0=-1, j0=-1;
    swap_grade = 0.0;

    return (int)swap_tracker.size();
}

void MaxVol::WriteData(std::ofstream & ofs)
{
    ofs.write((char*)&A(0, 0), A.size1*A.size2*sizeof(double));
    ofs.write((char*)&invA(0, 0), invA.size1*invA.size2*sizeof(double));
}

void MaxVol::ReadData(std::ifstream & ifs)
{
    ifs.read((char*)&A(0, 0), A.size1*A.size2*sizeof(double));
    ifs.read((char*)&invA(0, 0), invA.size1*invA.size2*sizeof(double));
    if (ifs.fail())
        ERROR("MaxVol::ReadData(): Error loading matrices");
}
