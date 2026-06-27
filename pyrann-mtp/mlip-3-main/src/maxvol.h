/*   This software is called MLIP for Machine Learning Interatomic Potentials.
 *   MLIP can only be used for non-commercial research and cannot be re-distributed.
 *   The use of MLIP must be acknowledged by citing approriate references.
 *   See the LICENSE file for details.
 *
 *   Contributors: Evgeny Podryabinkin
 */

#ifndef MLIP_MAXVOL_H
#define MLIP_MAXVOL_H

#include "common/stdafx.h"
#include "common/comm.h"
#include "common/multidimensional_arrays.h"


// Object for maximization of |det(A)| (volume) by swaping its rows with the rows of matrix B. Can maximize volume multiple times for a number of different B
class MaxVol  //: public LogWriting
{
private:
    Array2D invA;               // column-ordered (or A^(-T))
    Array2D BinvA;              // row-ordered 

    // temporal arrays
    Array1D bufVect1;       
    Array1D bufVect2;
    Array1D bufVect3;
    Array2D bufMtrx;

    int i0;                     // row index of maxmod element in B (0<=i0<=m)
    int j0;                     // column index of maxmod element in B (0<=j0<=n)
    double swap_grade;          // temporal for swap grade storage
#ifdef MLIP_MPI
    int mpi_swap_root;          // #process with the swapping row in B 
#endif

    double init_treshold;       // value for A initilization (A = IdentityMatrix * init_treshold)
    
    unsigned int eval_cntr = 0; // counts BinvA calculations

protected:
    static const char* tagname; // tag name of object
    int n = 0;                  // size of A is n x n. Read only

public:
    Array2D A;                  // Matrix of maximal volume 
    Array2D B;                  // Matrix which rows to maximize volume

    struct SwapStep
    {

        int ind_A;
        int ind_B;
#ifdef MLIP_MPI
        int mpi_root_B;         // #process with the swapping row in B 
        SwapStep(int _ind_B, int _ind_A, int _root_B) : 
            ind_A(_ind_A), ind_B(_ind_B), mpi_root_B(_root_B) {};
#else
        SwapStep(int _ind_B, int _ind_A) : ind_A(_ind_A), ind_B(_ind_B) {};
#endif 
    };
    std::vector<SwapStep> swap_tracker; // stores swap history for the last call of MaximizeVol(). The first is row index in B, the second is row index in A

    double log_n_det;           // stores logarithm of |det(A)| (of base n)

private:
    void UpdateInvA();
    void UpdateBinvA();
    void SwapRows();
#ifdef MLIP_MPI
    void MPI_ShareSwapData();   // FindMaxElem(...) finds the local swap data (max_grade, i0, j0, mpi_swap_root). This function sets the global data for all processors
#endif // MLIP_MPI
    
    void FindMaxElem(Array2D& Mtrx_, int start_row=0, int row_count=0);   // Finds maximal in modulus element in submatrix of Mtrx_ starting from 'start_row' to 'start_row'+'row_count'. Updates i0, j0, swap_grade
    void Iterate();

public:
    MaxVol();
    MaxVol(int _n, double _init_treshold = 1.0e-5);
    ~MaxVol();
    void Reset(int _n, double _init_treshold = 1.0e-5);          //!< Resets maxvol object 

    int InitedRowsCount();                                       //!< Returns the number of inited rows in A
    //void RecalculateInvA();                                    //!< Not actually required

    double CalcSwapGrade(   std::vector<int>* rows_ranges=nullptr,
                            std::vector<double>* out_grades=nullptr); //!< Calculates BinvA and finds array 'out_grades' of maximal-in-modulus elements in each submatrix of it (submatrices are defined by rows_ranges array). BinvA is sotored and can be used in MaximizeVol until MaximizeVol is called.
    int MaximizeVol(double threshold, int swap_limit=HUGE_INT);  //!< Main function maximizing |det(A)| by swapping its rows with rows of B. Does not calculate BinvA if CalcSwapGrade() called before (use BinvA calulated in CalcSwapGrade()).

    void WriteData(std::ofstream& ofs);                          //!< Writes A and invA in binary format. Required for Saving active learning
    void ReadData(std::ifstream& ifs);                           //!< Reads A and invA in binary format. Required for Saving active learning
};

#endif //MLIP_MAXVOL_H
