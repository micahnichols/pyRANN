/*   This software is called MLIP for Machine Learning Interatomic Potentials.
 *   MLIP can only be used for non-commercial research and cannot be re-distributed.
 *   The use of MLIP must be acknowledged by citing approriate references.
 *   See the LICENSE file for details.
 *
 *   This file contributors: Evgeny Podryabinkin
 */

#ifndef MLIP_ERROR_MONITOR_H
#define MLIP_ERROR_MONITOR_H

#include "common/comm.h"
#include "configuration.h"


// calculates difference in EFS for configurations. Used to calculate fitting errors
class ErrorMonitor : public InitBySettings
{
protected:
  static const char* tagname;                           // tag name of object

  void InitSettings()                                     // Sets correspondence between variables and setting names in settings file
  {
      MakeSetting(report_to, "report_to");
      MakeSetting(log, "log");
  };

public:
    class Quantity                                              // class storing some numbers related to errors 
    {
    public:
        double value = 0.0;                                     // exact value 
        double delta = 0.0;                                     // difference between exact and approximating value
        double valsq = 0.0;                                     // squared exact value
        double dltsq = 0.0;                                     // delta^2
        double reltv = 0.0;                                     // delta/value
        void clear() { value = delta = valsq = dltsq = reltv = 0.0; };
        Quantity() {};
    };

    class Accumulator                                           // class that accumulates quantities (Quantity) and stores maximal and sum of them
    {
    public:
        int count;                                              // number of accumulated quantities
        Quantity max;                                           // maximal of quantities
        Quantity sum;                                           // sum of quantities
        Accumulator() { clear(); };
        void accumulate_serial(Quantity& item, int count_increment=1)
        {
            max.value = __max(max.value, item.value);
            max.delta = __max(max.delta, item.delta);
            max.valsq = __max(max.valsq, item.valsq);
            max.dltsq = __max(max.dltsq, item.dltsq);
            max.reltv = __max(max.reltv, item.reltv);

            sum.value += item.value;
            sum.delta += item.delta;
            sum.valsq += item.valsq;
            sum.dltsq += item.dltsq;
            sum.reltv += item.reltv;

            count += count_increment;
        }
        void accumulate(Quantity& item, int count_increment=1)  // updates max and sum with item 
        {
#ifdef MLIP_MPI
            double buff_send[5];
            double buff_recv[5];
            buff_send[0] = item.value;
            buff_send[1] = item.delta;
            buff_send[2] = item.valsq;
            buff_send[3] = item.dltsq;
            buff_send[4] = item.reltv;

            MPI_Allreduce(buff_send, buff_recv, 5, MPI_DOUBLE, MPI_MAX, mpi.comm);

            max.value = __max(max.value, buff_recv[0]);
            max.delta = __max(max.delta, buff_recv[1]);
            max.valsq = __max(max.valsq, buff_recv[2]);
            max.dltsq = __max(max.dltsq, buff_recv[3]);
            max.reltv = __max(max.reltv, buff_recv[4]);

            MPI_Allreduce(buff_send, buff_recv, 5, MPI_DOUBLE, MPI_SUM, mpi.comm);

            sum.value += buff_recv[0];
            sum.delta += buff_recv[1];
            sum.valsq += buff_recv[2];
            sum.dltsq += buff_recv[3];
            sum.reltv += buff_recv[4];

            int buff_cnt;
            MPI_Allreduce(&count_increment, &buff_cnt, 1, MPI_INT, MPI_SUM, mpi.comm);

            count += buff_cnt;

#else
            accumulate_serial(item, count_increment);
#endif // MLIP_MPI
        };
        void accumulate(const Accumulator& item)                        // updates max and sum with item 
        {
#ifdef MLIP_MPI
            double buff_send[5];
            double buff_recv[5];
            buff_send[0] = item.max.value;
            buff_send[1] = item.max.delta;
            buff_send[2] = item.max.valsq;
            buff_send[3] = item.max.dltsq;
            buff_send[4] = item.max.reltv;

            MPI_Allreduce(buff_send, buff_recv, 5, MPI_DOUBLE, MPI_MAX, mpi.comm);

            max.value = __max(max.value, buff_recv[0]);
            max.delta = __max(max.delta, buff_recv[1]);
            max.valsq = __max(max.valsq, buff_recv[2]);
            max.dltsq = __max(max.dltsq, buff_recv[3]);
            max.reltv = __max(max.reltv, buff_recv[4]);

            buff_send[0] = item.sum.value;
            buff_send[1] = item.sum.delta;
            buff_send[2] = item.sum.valsq;
            buff_send[3] = item.sum.dltsq;
            buff_send[4] = item.sum.reltv;

            MPI_Allreduce(buff_send, buff_recv, 5, MPI_DOUBLE, MPI_SUM, mpi.comm);

            sum.value += buff_recv[0];
            sum.delta += buff_recv[1];
            sum.valsq += buff_recv[2];
            sum.dltsq += buff_recv[3];
            sum.reltv += buff_recv[4];

            int buff_cnt;
			int item_cnt = item.count;
            MPI_Allreduce(&item_cnt, &buff_cnt, 1, MPI_INT, MPI_SUM, mpi.comm);

            count += buff_cnt;

#else
            max.value = __max(max.value, item.max.value);
            max.delta = __max(max.delta, item.max.delta);
            max.valsq = __max(max.valsq, item.max.valsq);
            max.dltsq = __max(max.dltsq, item.max.dltsq);
            max.reltv = __max(max.reltv, item.max.reltv);

            sum.value += item.sum.value;
            sum.delta += item.sum.delta;
            sum.valsq += item.sum.valsq;
            sum.dltsq += item.sum.dltsq;
            sum.reltv += item.sum.reltv;

            count += item.count;
#endif // MLIP_MPI
        };
        void clear()                                            // Resets accumulated quntities
        {
            count = 0;
            max.clear();
            sum.clear();
        };
    };

    double relfrc_regparam = 0.0;   // regularization parameter for calculation of relative forces errors. If =0 absolute values are accumulated

    int cfg_count;                  // compared configuration counter

    // variables storing quantities for a pair configurations with exact and approximated EFS
    Quantity    ene_cfg;            // energy 
    Quantity    epa_cfg;            // energy per atom
    Quantity    str_cfg;            // stress
    Quantity    vir_cfg;            // virial stresses (in GPa)
    Accumulator frc_cfg;            // force

    // variables storing accumulated quantities
    Accumulator ene_all;            // energy 
    Accumulator epa_all;            // energy per atom
    Accumulator frc_all;            // stress
    Accumulator str_all;            // force
    Accumulator vir_all;            // virial stresses (in GPa)

    std::string log = "";
    std::string report_to = "stdout";

    ErrorMonitor(Settings setings, double _relfrc_regparam = 0.0);
    ErrorMonitor();
    ~ErrorMonitor();
    void Compare(Configuration& ValidCfg, Configuration& CheckCfg, double wgt=1.0); // compare two configurations updating ???_cfg
    void AddToCompare(Configuration& ValidCfg, Configuration& CheckCfg, double wgt=1.0); // compare and accumulate deviation data updating accumulators ???_all
    void GetReport(); // generate a report
    void Reset();                   // resets all accumulators

    inline double ene_rmsabs() { return sqrt(ene_all.sum.dltsq / (ene_all.count + 1.0e-300)); }     // root-mean-squred energy deviation
    inline double ene_aveabs() { return ene_all.sum.delta / (ene_all.count + 1.0e-300); }           // average energy deviation
    inline double ene_averel() { return ene_all.sum.reltv / (ene_all.count + 1.0e-300); }           // relative energy deviation
    inline double epa_rmsabs() { return sqrt(epa_all.sum.dltsq / (epa_all.count + 1.0e-300)); }     // root-mean-squred deviation of energy per atom
    inline double epa_aveabs() { return epa_all.sum.delta / (epa_all.count + 1.0e-300); }           // average deviation of energy per atom
    inline double epa_averel() { return epa_all.sum.reltv / (epa_all.count + 1.0e-300); }           // relative deviation of energy per atom
    inline double frc_aveabs() { return frc_all.sum.delta / (frc_all.count + 1.0e-300); }           // average absolute deviation of forces
    inline double frc_rmsabs() { return sqrt(frc_all.sum.dltsq / (frc_all.count + 1.0e-300)); }     // root-mean-squred deviation of forces
    inline double frc_averel() { return frc_all.sum.reltv / (frc_all.count + 1.0e-300); }           // average relative force error
    inline double frc_rmsrel() { return sqrt(frc_all.sum.dltsq / (frc_all.sum.valsq + 1.0e-300)); } // root-mean-squred deviation of forces divided by root-mean-squred force 
    inline double str_rmsabs() { return sqrt(str_all.sum.dltsq / (str_all.count + 1.0e-300)); }     // root-mean-squred stress deviation
    inline double str_rmsrel() { return sqrt(str_all.sum.dltsq / (str_all.sum.valsq + 1.0e-300)); } // root-mean-squred stress deviation divided by root-mean-squred  stress 
    inline double str_averel() { return str_all.sum.reltv / (str_all.count + 1.0e-300); }           // average relative stress 
    inline double str_aveabs() { return str_all.sum.delta / (str_all.count + 1.0e-300); }           // average absolute stress 
    inline double vir_rmsabs() { return sqrt(vir_all.sum.dltsq / (vir_all.count + 1.0e-300)); }     // root-mean-squred viriral stress deviation
    inline double vir_rmsrel() { return sqrt(vir_all.sum.dltsq / (vir_all.sum.valsq + 1.0e-300)); } // root-mean-squred viriral stress deviation divided by root-mean-squred  stress 
    inline double vir_averel() { return vir_all.sum.reltv / (vir_all.count + 1.0e-300); }           // average relative viriral stress 
    inline double vir_aveabs() { return vir_all.sum.delta / (vir_all.count + 1.0e-300); }           // average absolute viriral stress 
};

#endif //#ifndef MLIP_ERROR_MONITOR_H

