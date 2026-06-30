# pyRANN

A python package for generating atomic structures for machine learned interatomic potentials (MLIPs), visualizing and refining MLIP databases and metaparameters, and general MLIP tools

To enable calculation of MTP atomic descriptors, make sure to install the optional pyrann[mtp] package along with pyrann.

For documentation, please see pyrann.readthedocs.io/en/latest/pyrann.html

## Installation

Download or clone this repository and then run the following:

```bash
cd pyRANN
pip3 install ".[all]"
```

## Documentation

Please visit the html documentation in /docs/build/html/pyrann.html as well as the tutorial in /docs/build/html/tutorial.html

## Tutorial

This is a tutorial for plotting the descriptor space of a sample MLIP database (Mg) with RANN and MTP atomic descriptors. Please open /pyrann/docs/build/html/tutorial.html in a browser to see the full interactive plots. Alternatively, open the plots themselves in /pyrann/docs/source/_static/*.html

```python

import pyrann as rann
import numpy as np

# Create processing class instance with Mg RANN input file
processing = rann.processing('Mg.input')

# Set processing instance features to RANN potential atomic descriptors
# Because the formalism is RANN, it also sets the Global Simulation Number,
#   Local Simulation Number, and Local ID for each atom
processing.get_features()

# Initialize UMAP instance for processing instance and return mapper
 mapper = processing.get_umap()

# Plot UMAP embedding of atomic descriptor space with each atom colored by
#   filename and rename the plot to "full_database.html"
processing.plot(mapper, color_data=processing.filenames,
                hover_info={'Global Simulation Number': processing.global_sim_num,
                            'Local Simulation Number': processing.local_sim_num,
                            'Local ID': processing.local_id},
                key_str='full_database')

# Reducing the database by 50%
reduced = processing.reduce(percent=50.0)

# Creating a boolean mask to only plot the reduced database
subset_points = [True if i in reduced else False for i in processing.global_sim_num]

# Plot UMAP embedding of atomic descriptor space of the reduced database with
#   each atom colored byfilename and rename the plot to "reduced_database.html"
processing.plot(mapper, color_data=processing.filenames,
                hover_info={'Global Simulation Number': processing.global_sim_num,
                            'Local Simulation Number': processing.local_sim_num,
                            'Local ID': processing.local_id},
                subset_points=subset_points,
                key_str='reduced_database')


# Get unique filenames in the order that they are read for feature calculation
unique_filenames, unique_index = np.unique(processing.filenames,
                                           return_index=True)
unique_index = np.argsort(unique_index)
unique_filenames = unique_filenames[unique_index]

# Load dump files to a list of series instances in the order they are read for
#   feature calculation
series = [rann.load(i, series=True) for i in unique_filenames]

# Get new series instances for reduced database
global_count = 0
reduced_series = []
for i in range(len(unique_filenames)):
  file_series = []
  for j in range(len(series[i].systems)):
    if global_count in reduced:
      file_series.append(series[i].systems[j])
    global_count += 1
  reduced_series.append(rann.series(file_series))

# Export reduced database to new dump files
for i in range(len(reduced_series)):
  reduced_series[i].export(f'{unique_filenames[i]}_reduced.dump',
                           directory='reduced_DUMPS')

# Getting the energy per atom for each atom for each simulation
epa = [j.energy / j.natoms for i in series for j in i.systems for _ in range(j.natoms)]

# Plotting the full database with atoms colored by energy per atom (eV/atom)
processing.plot(mapper, color_data=epa,
                color_type='linear',
                hover_info={'Global Simulation Number': processing.global_sim_num,
                            'Local Simulation Number': processing.local_sim_num,
                            'Local ID': processing.local_id,
                            'eV/atom': epa},
                key_str='full_database_epa')

# Plotting the reduced database with atoms colored by energy per atom (eV/atom)
processing.plot(mapper, color_data=epa,
                color_type='linear',
                hover_info={'Global Simulation Number': processing.global_sim_num,
                            'Local Simulation Number': processing.local_sim_num,
                            'Local ID': processing.local_id,
                            'eV/atom': epa},
                subset_points=subset_points,
                key_str='reduced_database_epa')
```

To reprduce the same plots on the same database using MTP descriptors, only lines 5 and 10 need to change.

```python
# Create processing class instance with Mg MTP potential
processing = rann.processing('pot4_6.0.mtp')

# Set processing instance features to MTP potential atomic descriptors
#   Setting file_fmt to 'dump' will convert all dump files in the directory
#   to cfg files in order to calculate MTP descriptors
processing.get_features(file_fmt='dump')
```
