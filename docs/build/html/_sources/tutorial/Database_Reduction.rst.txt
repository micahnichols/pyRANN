Database Reduction using pyRANN
===============================

We can reduce the MLIP database using pyRANN and the atomic descriptors.

pyRANN reduces the database by the following steps: 1) Calculate atomic
descriptors 2) Determine the 100 nearest neighbors for each atom in
descriptor space - Neighbors are considered valid if the neighbor is not
in the same simulation as the parent atom and if the simulation the
neigbor atom belongs to has not already been deleted 3) Calculate the
neighbor density for each atom. Density is taken to be 1 / average
neighbor distance of all valid neighbors. If there are no valid
neighbors, the average distance is set to an arbitrarily high number 4)
Calculate the average neighbor density per simulation 5) Delete the
simulation with the highest density. This is done by marking the global
simulation number as deleted and no longer valid 6) Do steps 2-5 for
:math:`int(\text{number of global simulations} * (\frac{\text{percent of database to delete}}{100}))`
total steps

First we need to calculate the atomic descriptors. For more help with
this, please see the atomic descriptors tutorial. For the sake of
brevity, this tutorial will only reduce the database using the RANN
formalism.

.. code:: ipython3

    import numpy as np
    import pyrann as rann
    
    rann_pot = rann.processing('Mg.input')
    rann_pot.get_features()
    print(rann_pot.features)


.. parsed-literal::

    Initializing pyrann version 1.0.0.a1
    
    # Number Threads     : 16
    [[ 3.38935976  3.02644257  2.25109709 ...  1.61972099  1.95561874
       1.68242089]
     [ 6.37313423  5.05768164  3.35657445 ...  8.02213011  8.21899046
       8.04288521]
     [ 1.16156767  1.51729725  1.84310613 ... -0.03339917  0.02580062
      -0.024193  ]
     ...
     [-1.07721352 -1.49743447 -1.95603472 ... -0.04121257 -0.11339892
      -0.04301757]
     [-0.32591448 -0.50494778 -0.74019567 ... -0.03560158 -0.06744371
      -0.03037547]
     [-0.6583386  -0.85380547 -1.04831626 ... -0.05904069 -0.10222123
      -0.05760668]]


Let’s plot the original database first for comparison after we reduce.

.. code:: ipython3

    rann_mapper = rann_pot.get_umap()
    rann_pot.plot(rann_mapper, color_data=rann_pot.filenames,
                  hover_info={'Global Simulation Number': rann_pot.global_sim_num,
                              'Local Simulation Number': rann_pot.local_sim_num,
                              'Local ID': rann_pot.local_id},
                  key_str='rann_original')


.. parsed-literal::

    2026-06-27 12:15:04.467326: I tensorflow/core/util/port.cc:153] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
    2026-06-27 12:15:04.527616: E external/local_xla/xla/stream_executor/cuda/cuda_fft.cc:467] Unable to register cuFFT factory: Attempting to register factory for plugin cuFFT when one has already been registered
    WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
    E0000 00:00:1782580504.559538   15258 cuda_dnn.cc:8579] Unable to register cuDNN factory: Attempting to register factory for plugin cuDNN when one has already been registered
    E0000 00:00:1782580504.567512   15258 cuda_blas.cc:1407] Unable to register cuBLAS factory: Attempting to register factory for plugin cuBLAS when one has already been registered
    W0000 00:00:1782580504.626671   15258 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
    W0000 00:00:1782580504.626733   15258 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
    W0000 00:00:1782580504.626735   15258 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
    W0000 00:00:1782580504.626736   15258 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
    2026-06-27 12:15:04.639430: I tensorflow/core/platform/cpu_feature_guard.cc:210] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
    To enable the following instructions: AVX2 AVX512F AVX512_VNNI FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
    /home/micah/python-env/lib/python3.12/site-packages/sklearn/utils/deprecation.py:151: FutureWarning: 'force_all_finite' was renamed to 'ensure_all_finite' in 1.6 and will be removed in 1.8.
      warnings.warn(


.. parsed-literal::

    UMAP(init='pca', n_neighbors=54, verbose=True)
    Sat Jun 27 12:15:07 2026 Construct fuzzy simplicial set
    Sat Jun 27 12:15:07 2026 Finding Nearest Neighbors
    Sat Jun 27 12:15:07 2026 Building RP forest with 20 trees
    Sat Jun 27 12:15:10 2026 NN descent for 16 iterations
    	 1  /  16
    	 2  /  16
    	Stopping threshold met -- exiting after 2 iterations
    Sat Jun 27 12:15:28 2026 Finished Nearest Neighbor Search
    Sat Jun 27 12:15:30 2026 Construct embedding



.. parsed-literal::

    Epochs completed:   0%|            0/200 [00:00]


.. parsed-literal::

    	completed  0  /  200 epochs
    	completed  20  /  200 epochs
    	completed  40  /  200 epochs
    	completed  60  /  200 epochs
    	completed  80  /  200 epochs
    	completed  100  /  200 epochs
    	completed  120  /  200 epochs
    	completed  140  /  200 epochs
    	completed  160  /  200 epochs
    	completed  180  /  200 epochs
    Sat Jun 27 12:15:40 2026 Finished embedding


.. code:: ipython3

    from IPython.display import IFrame
    
    IFrame(src='rann_original.html', width='100%', height=1000)




.. raw:: html

    
    <iframe
        width="100%"
        height="1000"
        src="../_static/rann_original.html"
        frameborder="0"
        allowfullscreen
    
    ></iframe>




Now, we will reduce the database by 50%

.. code:: ipython3

    reduced = rann_pot.reduce(percent=50.0)

The reducd method of the processing class returns an array containing
the global simulation numbers that were **not** deleted. These
simulations make up the reduced database. It is possible to plot only
the reduced database using the UMAP mapper that was fit on the full
database. It is also possible to plot the reduced database with a new
UMAP mapper by setting the features object equal to only the features of
the simulations in the reduced database.

Here we will plot the features using the original UMAP mapper as it is a
better comparison method. In order to do so, we need to create a boolean
array to pass to subset_points argument of the plot method. This boolean
array should be of len(global_sim_num) and have a value of True if the
global simulation is in the reduced database and a value of False
otherwise.

.. code:: ipython3

    subset_points = np.array([True if i in reduced else False for i in rann_pot.global_sim_num])
    
    rann_pot.plot(rann_mapper, color_data=rann_pot.filenames,
                  subset_points=subset_points,
                  hover_info={'Global Simulation Number': rann_pot.global_sim_num,
                              'Local Simulation Number': rann_pot.local_sim_num,
                              'Local ID': rann_pot.local_id},
                  key_str='rann_reduced')
    
    IFrame(src='rann_reduced.html', width='100%', height=1000)




.. raw:: html

    
    <iframe
        width="100%"
        height="1000"
        src="../_static/rann_reduced.html"
        frameborder="0"
        allowfullscreen
    
    ></iframe>




Comparing the two plots, the region of FCC thermally perturbed data is
the most effected. Reducing dump files that are clustered together in
the full figure may be a better approach for large databases as it looks
like the HCP thermally perturbed region that also contains tetrahedral
interstitial data can be reduced much further and still span the same
space. A more quantitative and robust way to compare the space spanned
is currently under development.

