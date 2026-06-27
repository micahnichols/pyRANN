/*   MLIP is a software for Machine Learning Interatomic Potentials
 *   MLIP is released under the "New BSD License", see the LICENSE file.
 *   Contributors: Evgeny Podryabinkin
 */


#include "../wrapper.h"


using namespace std;


// Initilizes MLIP
#ifndef MLIP_MPI
#   define MPI_Comm int
#endif


void* MLIP_init(std::map<std::string, std::string> setup,  // settings filename
                double& rcut,                // MLIP's cutoff radius returned to LAMMPS (may be used for parallelization or acceleration)
                MPI_Comm& world,
                void (*CallbackComm)(void*, double*))
{
    Wrapper* MLIP_wrp = nullptr;

    mpi.InitComm(world);

    mpi.LammpsCallbackComm = CallbackComm;

    SetStreamForOutput(&std::cout);

    Settings settings(setup);

    try
    {
        MLIP_wrp = new Wrapper(settings);
        AnyLocalMLIP* p_mlip = (AnyLocalMLIP*)MLIP_wrp->p_mlip;
        if (p_mlip == nullptr)
            ERROR("Incoorect MLIP initialization");

        rcut = p_mlip->CutOff();

        return static_cast<void*>(MLIP_wrp); 
    }
    catch (MlipException& exception)
    {
        Message(exception.What());
        std::cout.flush();
        exit(9991);
    }
}

void MLIP_CalcCfgForLammps( void* p_void_mlip,      // a pointer to mlip_wrapper stored in lammps
                            void* p_void_pair,      // a pointer to lammps pair object calling this function (required for callback from mlip)
                            int inum,               // input parameter: number of neighborhoods (number of local atoms)
                            int nghost,             // number of ghost atoms
                            int* ilist,                          // input parameter: 
                            int* numneigh,                      // input parameter: number of neighbors in each neighborhood (inum integer numbers)
                            int** firstneigh,                  // input parameter: pointer to the first neighbor
                            double* lattice,                  // input parameter: lattice (9 double numbers)
                            double** x,                      // input parameter: array of coordinates of atoms
                            int* types,                     // input parameter: array of atom types (inum of integer numbers)
                            bool splited,
                            double** f,                    // output parameter: forces on atoms (cartesian, n x 3 double numbers)
                            double& en,                    // output parameter: summ of site energies 
                            double* stresses,              // output parameter: stresses in eV (9 double numbers)
                            double* site_en=nullptr,       // output parameter: array of site energies (inum double numbers). if =nullptr while call no site energy calculation is done
                            double** site_virial=nullptr)  // output parameter: array of site energies (inum double numbers). if =nullptr while call no virial-stress-per-atom calculation is done
{
    try
    {
        Wrapper* MLIP_wrp = static_cast<Wrapper*>(p_void_mlip);

        if (MLIP_wrp == nullptr)
            ERROR("A pointer to the MLIP Wrapper is nullptr");

        const size_t NEIGHMASK = 0x3FFFFFFF;
        Configuration cfg;
        cfg.p_void_pair = p_void_pair;

//        static int cfg_counter = 0;
//        cfg.features["ind"] = to_string(++cfg_counter);

        double cutoff = ((AnyLocalMLIP*)MLIP_wrp->p_mlip)->CutOff();

#   ifdef MLIP_DEBUG
        if (mpi.size == 1 && cfg.is_mpi_splited == true)
            ERROR("Configuration state is inconsistent with MPI_SIZE");

        int splited_everywhere;
        int splited_check = cfg.is_mpi_splited ? 1 : 0;
        MPI_Allreduce(&splited_check, &splited_everywhere, 1, MPI_INT, MPI_SUM, mpi.comm);
        if (splited_everywhere != 0 && splited_everywhere != mpi.size)
            ERROR("Contradictory configuration status");
#   endif

        // set lattice 
        int foo = 0;
        for (int a=0; a<3; a++)
            for (int b=0; b<3; b++)
                cfg.lattice[a][b] = lattice[foo++];

        //cout << "inum = " << inum << ", nghost = " << nghost << endl;
    
        cfg.resize(inum + nghost);
        cfg.ClearNbhs();                    // (!) required to destroy the arrays of nbh from previous data
        cfg.nbhs.resize(inum);              // nbhs array is created for all atoms, whereas nbh structures are initilized only for non-ghost atoms and remain empty for ghost neighborhoods
        cfg.is_mpi_splited = splited;

        memcpy(&cfg.pos(0, 0), x[0], 3 * (inum + nghost) * sizeof(double));
        memcpy(&cfg.type(0), types, (inum + nghost) * sizeof(int));
        for (int i=0; i<inum+nghost; i++)
            cfg.type(i) -= 1;

        // constructing neighborhoods
        for (int ii=0; ii<inum; ii++)
        {
            int i = ilist[ii];
            double xtmp = x[i][0];
            double ytmp = x[i][1];
            double ztmp = x[i][2];
            int* jlist = firstneigh[i];
            int jnum = numneigh[i];

            Neighborhood& nbh = cfg.nbhs[ii];

            // Constructing neighborgood
            nbh.count = 0;
            nbh.my_type = types[i]-1;
            nbh.my_ind = i;

            for (int jj=0; jj<jnum; jj++)
            {
                int j = jlist[jj];
                j &= NEIGHMASK;

                double delx = x[j][0] - xtmp;
                double dely = x[j][1] - ytmp;
                double delz = x[j][2] - ztmp;
                double r = sqrt(delx*delx + dely*dely + delz*delz);

                if (r < cutoff)
                {
                    nbh.count++;
                    nbh.inds.emplace_back(j);
                    nbh.vecs.emplace_back(delx, dely, delz);
                    nbh.dists.emplace_back(r);
                    nbh.types.emplace_back(types[j]-1);
                }
                // else nbh remain empty with nbh.count=0 (nbh.count=0 is an indicator of ghost neighborhood)
            }
        }
        cfg.nbh_cutoff = cutoff;

        // Gathering indices of ghost atoms
        vector<bool> is_not_ghost(inum+nghost, false);
        for (int ii=0; ii<inum; ii++)
            is_not_ghost[ilist[ii]] = true;
        for (int i=0; i<inum+nghost; i++)
            if (!is_not_ghost[i])
                cfg.ghost_inds.insert(i);

        MLIP_wrp->Process(cfg);
            if (site_en != nullptr)
                if (MLIP_wrp->p_mlip != nullptr)
                    for (int i=0; i<cfg.nbhs.size(); i++)
                    {
                        Neighborhood& nbh = cfg.nbhs[i];
                        site_en[i] = ((AnyLocalMLIP*)MLIP_wrp->p_mlip)->SiteEnergy(nbh);
                    }

        en = cfg.energy;

        if (cfg.has_forces())
            //memcpy(f[0], &cfg.force(0, 0), 3*(inum+nghost)*sizeof(double));
            for (int i = 0; i < (inum + nghost); i++) {
                for (int j = 0; j < 3; j++) {
                    f[i][j] += cfg.force(i,j);
                }
            } 
        else if (cfg.size() > 0)
            ERROR("Forces have not been calculated by mlip for some reason");

        foo = 0;
        for (int a=0; a<3; a++)
            for (int b=0; b<3; b++)
                stresses[foo++] = cfg.stresses[a][b];
    }
    catch (MlipException& exception)
    {
        string mess;
#ifdef MLIP_MPI
        if (((std::string)exception.What()).size() > 0)
            mess += "Rank " + to_string(mpi.rank) + ", ";
#endif
        mess += exception.What();
        cerr << mess << endl;
        exit(777);
    }
}


// destroys MLIP object
void MLIP_finalize(void* p_void_mlip)
{
    Wrapper* MLIP_wrp = static_cast<Wrapper*>(p_void_mlip);

    try
    {
        if (MLIP_wrp != nullptr)
            delete MLIP_wrp;
        p_void_mlip = nullptr;
    }
    catch (MlipException& excp)
    {
        Message(excp.What());
        exit(9994);
    }
}

