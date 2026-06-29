pyRANN API
===============
.. automodule:: pyrann

.. autosummary::
   :signatures: short 

   load
   system
   series
   calibration
   processing

.. autofunction:: load

.. automodule:: pyrann.system

.. autoclass:: system
   :members:
   :member-order: groupwise
   :undoc-members:
   :exclude-members: findspacing, makelattice3, thermal_test, thermal_wave

   .. rubric:: Attributes

   .. autosummary::
      :signatures: short

      ~atoms
      ~box
      ~descriptor
      ~elements
      ~energy
      ~force
      ~natoms
      ~stress
      ~timestep
      ~types

   .. rubric:: Methods

   .. autosummary::
      :signatures: short

      ~change_type
      ~createsupercell
      ~export
      ~gsfe
      ~solidsolution
      ~strain
      ~surface
      ~thermal
      ~vacancy

.. automodule:: pyrann.series

.. autoclass:: series
   :members:
   :member-order: groupwise
   :undoc-members:
   :show-inheritance:

   .. rubric:: Attributes

   .. autosummary::
      :signatures: short

      ~series.descriptor
      ~series.systems
      ~series.timesteps

   .. rubric:: Methods

   .. autosummary::
      :signatures: short

      ~series.export

.. automodule:: pyrann.processing

.. autoclass:: processing
   :members:
   :member-order: groupwise
   :undoc-members:
   :show-inheritance:

   .. rubric:: Attributes

   .. autosummary::
      :signatures: short

      ~processing.features
      ~processing.formalism
      ~processing.global_sim_num
      ~processing.input_file
      ~processing.local_id
      ~processing.local_sim_num


   .. rubric:: Methods

   .. autosummary::
      :signatures: short

      ~processing.density
      ~processing.get_features
      ~processing.get_umap
      ~processing.plot
      ~processing.reduce

calibration
-------------------
.. automodule:: pyrann.calibration

   .. rubric:: Class

   .. autosummary::
      :signatures: short

      ~PairRANN
      ~Simulation

.. autoclass:: PairRANN
   :members:
   :member-order: groupwise
   :undoc-members:
   :show-inheritance:

   .. rubric:: Attributes

   .. autosummary::
      :signatures: short

      ~PairRANN.natoms
      ~PairRANN.nsims
      ~PairRANN.sims_per_file

   .. rubric:: Methods

   .. autosummary::
      :signatures: short

      ~PairRANN.feature
      ~PairRANN.finish
      ~PairRANN.firstneigh
      ~PairRANN.force
      ~PairRANN.id
      ~PairRANN.run
      ~PairRANN.setup
      ~PairRANN.types

.. autoclass:: Simulation
   :members:
   :member-order: groupwise
   :undoc-members:
   :show-inheritance:
      
   .. rubric:: Attributes

   .. autosummary::
      :signatures: short

      ~Simulation.energy
      ~Simulation.filename
      ~Simulation.inum
      ~Simulation.types

   .. rubric:: Methods

   .. autosummary::
      :signatures: short

      ~Simulation.feature
