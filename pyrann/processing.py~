from __future__ import annotations
import numpy as np
import numpy.typing as npt
from typing import Union, Optional
from collections.abc import Callable
from . import calibration
from .load import load
from .system import system
from .series import series
import pandas as pd
import os
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from bokeh.models import Legend, LegendItem, LinearColorMapper, ColorBar
from bokeh.palettes import Viridis256
import bokeh.plotting as bpl
from bokeh.transform import transform
import matplotlib.colors as plc
import matplotlib.pyplot as plt
import colorcet.plotting
from bokeh.models import RangeSlider, CustomJS
from bokeh.layouts import column
import importlib.util

class processing:
    """
    Initializes a processing class instance

    Args:
        input_file: The name (relative path) of the file that contains the ML
            potential information.
        formalism: The formalism of the ML potential under
            investigation. Current options are RANN and MTP. If not provided,
            it tries to infer based on input_file extension.
    """

    def __init__(self,
                 input_file: str,
                 formalism: Optional[Union[str, None]] = None):
        relative_base = Path('.')
        files = list(relative_base.rglob(input_file))
        if Path(input_file) not in files:
            raise ValueError('Cannot find the provided input_file. Please provide the relative path.')
        else:
            self.input_file = input_file
        if formalism is None:
            input_ext = input_file.lower().split('.')[-1]
            if input_ext == 'nn' or input_ext == 'rann' or input_ext == 'input':
                formalism = 'rann'
            if input_ext == 'mtp':
                formalism = 'mtp'
        else:
            formalism = formalism.lower()
            formalisms = ['rann', 'nn', 'mtp']
            if formalism not in formalisms:
                raise ValueError('The provided formalism is not currently accepted. Accepted formalisms are "RANN" and "MTP".')
        self.formalism = formalism
        self.features = np.zeros((100,100))
        self.global_sim_num = np.zeros(100)

    @property
    def input_file(self) -> str:
        """
        Returns a string representation of the input file.

        Returns:
            A string representation of the input file.
        """
        return self.__input_file

    @input_file.setter
    def input_file(self, value: str):
        """
        Sets the input file for the processing class.

        Args:
            value (str): Relative path of the input file.

        Raises:
            ValueError: If the input file cannot be found.
        """
        relative_base = Path('.')
        files = list(relative_base.rglob(value))
        if Path(value) not in files:
            raise ValueError('Cannot find the provided input_file. Please provide the relative path.')
        else:
            self.__input_file = value

    @property
    def formalism(self) -> str:
        """
        Returns the current formalism.

        Returns:
            The current formalism.
        """
        return self.__formalism

    @formalism.setter
    def formalism(self, value: Union[str, None] = None):
        """
        Sets the formalism for the processing class.

        Args:
            value (str): The formalism of the ML potential under
            investigation. Current options are RANN and MTP. If not provided,
            it tries to infer based on input_file extension.

        Raises:
            ValueError: If the provided formalism is not accepted. Currently
            accepted formalisms are "RANN" and "MTP".
        """
        if value is None:
            input_ext = self.input_file.lower().split('.')[-1]
            if input_ext == 'nn' or input_ext == 'rann':
                formalism = 'rann'
            if input_ext == 'mtp':
                formalism = 'mtp'
        else:
            formalism = value.lower()
            formalisms = ['rann', 'nn', 'mtp']
            if formalism not in formalisms:
                raise ValueError('The provided formalism is not currently accepted. Accepted formalisms are "RANN" and "MTP".')
        self.__formalism = formalism

    @property
    def features(self) -> np.ndarray:
        """
        Returns the features

        Returns:
            np.ndarray: features of shape (natoms, nfeatures)
        """
        return self.__features
    @features.setter
    def features(self, value: npt.ArrayLike):
        """
        Sets the features for the class. Features should be of shape
        (natoms, nfeatures) where natoms is the number of atoms in 
        the database and nfeatures is the length of each feature
        vector.

        Args:
            value: Features array of shape (natoms, nfeatures).
        """
        self.__features = value

    @property
    def global_sim_num(self) -> np.ndarray:
        """
        Returns array of the global simulation number each atom
        belongs to.

        Returns:
            np.ndarray: 1-D array of global simulation numbers
            each atom belongs to.
        """
        return self.__global_sim_num

    @global_sim_num.setter
    def global_sim_num(self, value: npt.ArrayLike):
        """
        Sets the global simulation number for each atom

        Args:
            value: 1-D array of global simulation numbers
            that each atom belongs to
        """
        self.__global_sim_num = value

    def get_features(self,
                     standardize: Optional[bool] = True,
                     file_fmt: Optional[Union[str, None]] = None
                     ):
        """
        Gets and sets the filenames, global simulation numbers, local simulation numbers,
        atom ids, and features for the current formalism and input_file.

        Args:
            standardize: True if the features need to be standardized using
                StandardScalar
            file_fmt: File format for the structures to be investigated. 
                Only neccessary if the file format does not match the default
                file format of the current formalism. The files will be 
                converted to support the current set formalism.

        Sets:
            features (np.ndarray): An array of size (natoms, nfeatures). 
                Gives the features for all atoms.
        """

        path = os.getcwd()
        if file_fmt:
            fmts = ['dump', 'cfg', 'dat', 'data', 'poscar']
            if file_fmt.lower() not in fmts:
                raise ValueError('Supported formats are cfg, LAMMPS data, LAMMPS dump, and poscar')
            else:
                # series_list = []
                if file_fmt.lower() == 'dump' or file_fmt.lower() == 'cfg':
                    series_bool = True
                else:
                    series_bool = False
                if self.formalism == 'rann':
                    for file in os.listdir(path):
                        if file.lower().endswith(file_fmt.lower()):
                            temp = load(file, series=series_bool)
                            temp.export(f'{file.split('.')[0]}.dump')
                elif self.formalism == 'mtp':
                    temp_list = []
                    for file in os.listdir(path):
                        if file.lower().endswith(file_fmt.lower()):
                            temp = load(file, series=series_bool)
                            temp_list.append(temp)
                            temp.export(f'{file.split('.')[0]}.cfg')
                    # export = series([j for i in range(len(temp_list)) for j in temp_list[i].systems])
                    # export.export('combined.cfg')

                # TODO - add in MTP support
        if self.formalism == 'rann':
            a = calibration.PairRANN(self.input_file)
            a.setup()
            filenames = np.array([a[i].filename for i in range(a.nsims)
                                  for j in range(a[i].inum)])
            global_sim_num = np.array([i for i in range(a.nsims) for j in range(a[i].inum)])
            local_temp = np.array([j for i in a.sims_per_file for j in range(i)], dtype=np.int64)
            local_sim_num = np.array([local_temp[i] for i in range(a.nsims) for j in range(a[i].inum)], dtype=np.int64)
            local_id = np.array([a.id(i)[j] for i in range(a.nsims) for j in range(a[i].inum)], dtype=np.int64)
            features = np.array([a.feature(i,j) for i in range(a.nsims) for j in range(a[i].inum)], dtype=np.float64)

            self.pair_rann = a

        elif self.formalism == 'mtp':
            if not importlib.util.find_spec('pyrann.mtp_bindings'):
                raise ImportError('\nPlease compile mtp_bindings shared library file.\n')
            else:
                from . import mtp_bindings
                cfgs = []
                filename_list = []
                loaded_cfgs = []
                for file in os.listdir(path):
                    if file.lower().endswith('.cfg'):
                        loaded_cfgs.append(load(file, series=True))
                        filename_list.append(file)
                        cfgs.append(mtp_bindings.Configuration(file))
                pot = mtp_bindings.MTP(self.input_file)
                cutoff = pot.cutoff()
                for i in cfgs:
                    i.init_nbhs(cutoff)
                if len(cfgs) == 1:
                    features = np.array([pot.calc_basis_funcs(k) for i in cfgs for j in range(i.ncfg()) for k in i.nbhs(j)]) 
                    filenames = np.array([filename_list[i] for i in range(len(filename_list)) for j in range(len(loaded_cfgs[i].systems)) for k in range(loaded_cfgs[i].systems[j].natoms)])
                    local_sim_num = np.array([j for i in range(len(filename_list)) for j in range(len(loaded_cfgs[i].systems)) for k in range(loaded_cfgs[i].systems[j].natoms)])
                    # Working under the assumption that MTP trains over only 1 .cfg file
                    global_sim_num = np.array([j for i in range(len(filename_list)) for j in range(len(loaded_cfgs[i].systems)) for k in range(loaded_cfgs[i].systems[j].natoms)])
                    local_id = np.array([k for i in range(len(filename_list)) for j in range(len(loaded_cfgs[i].systems)) for k in range(loaded_cfgs[i].systems[j].natoms)])
                else:
                    features = np.array([pot.calc_basis_funcs(k) for i in cfgs for j in range(i.ncfg()) for k in i.nbhs(j)]) 
                    filenames = np.array([filename_list[i] for i in range(len(filename_list)) for j in range(len(loaded_cfgs[i].systems)) for k in range(loaded_cfgs[i].systems[j].natoms)])
                    local_sim_num = np.array([j for i in range(len(filename_list)) for j in range(len(loaded_cfgs[i].systems)) for k in range(loaded_cfgs[i].systems[j].natoms)])
                    # global_temp = 0
                    # TODO - MAKE SURE THIS MATH IS CORRECT
                    # global_sim_num = np.array([global_temp:=global_temp*i+(j+i) for i in range(len(filename_list)) for j in range(len(loaded_cfgs[i].systems)) for k in range(loaded_cfgs[i].systems[j].natoms)])
                    global_sim_num = []
                    global_temp = 0
                    for i in range(len(filename_list)):
                        for j in range(len(loaded_cfgs[i].systems)):
                            for k in range(loaded_cfgs[i].systems[j].natoms):
                                global_sim_num.append(global_temp)
                            global_temp += 1
                    global_sim_num = np.asarray(global_sim_num)

                    local_id = np.array([k for i in range(len(filename_list)) for j in range(len(loaded_cfgs[i].systems)) for k in range(loaded_cfgs[i].systems[j].natoms)])

        self.filenames = filenames
        self.global_sim_num = global_sim_num
        self.local_sim_num = local_sim_num
        self.local_id = local_id




        if standardize:
            scaler = StandardScaler()
            scaler.fit(features)
            features = scaler.transform(features)
        self.features = features

    def density(self) -> np.ndarray:
        """
        Finds the average feature-space density for each simulation using the
        100 nearest neighbors for each atom. Excludes atoms that belong to the
        same global simulation number.

        Returns:
            np.ndarray: 1-D array (length nsims) of average feature-space density.

        Raises:
            ValueError: If either features or global_sim_num is not set
        """
        # if self.features == np.zeros((100,100)) or self.global_sim_num == np.zeros(100):
        if not np.any(self.features) or not np.any(self.global_sim_num):
            raise ValueError('Please set features and global_sim_num before calling this function.')

        model = NearestNeighbors(n_neighbors=100)
        model.fit(self.features)
        dd, ii = model.kneighbors(self.features)

        global_unique, global_index = np.unique(self.global_sim_num, return_index=True)
        global_unique = self.global_sim_num[global_index]

        n_points = len(ii)
        k = ii.shape[1]

        neighbor_sims = global_sim[ii]
        same_sim = (neighbor_sims == global_sim[:, None])
        valid = (~same_sim)

        has_valid = valid.any(axis=1)

        rows = np.arange(n_points)

        fallback_idx = k-1

        avg_dist = np.where(
                has_valid,
                np.mean(np.ma.masked_array(dd[rows, :], mask=~valid), axis=1),
                1000*dd[rows, fallback_idx]
                )
        dens = 1.0 / avg_dist
        return dens

    def reduce(self,
               percent: float = 50.0,
               metric: Optional[Union[Callable, None]] = None,
               **kwargs
               ) -> np.ndarray:
        """
        Reduces the database. The default metric is to delete simulations in
        the most dense regions of the fingerprint space.

        Args:
            percent: The percentage of data to delete.
            metric: If given, this must be a user defined function that takes
                in values and deletes a certain amount of data.
            **kwargs: Any additional arguments that the user defined function
                needs.

        Returns:
            np.ndarray: 1-D array of global simulation numbers to keep in the database.
        """
        # loop_size = int(len(self.global_sim_num)*(percent/100))
        # print(f'{loop_size = }')

        if metric == None:

            deleted = np.zeros(len(self.global_sim_num))
            model = NearestNeighbors(n_neighbors=100)
            model.fit(self.features)
            dd, ii = model.kneighbors(self.features)

            global_unique, global_index = np.unique(self.global_sim_num, return_index=True)
            global_unique = self.global_sim_num[global_index]
            loop_size = int(len(global_unique)*(percent/100))

            n_points = len(ii)
            k = ii.shape[1]
            for i in range(loop_size):
                neighbor_sims = self.global_sim_num[ii]

                same_sim = (neighbor_sims == self.global_sim_num[:, None])
                nbr_deleted = deleted[neighbor_sims]

                valid = (~same_sim) & (nbr_deleted == 0)
                has_valid = valid.any(axis=1)
                rows = np.arange(n_points)
                fallback_idx = k-1
                avg_dist = np.where(
                        has_valid,
                        np.mean(np.ma.masked_array(dd[rows, :], mask=~valid), axis=1),
                        1000*dd[rows, fallback_idx]
                        )
                dens = 1.0 / avg_dist

                metric = np.where(deleted[global_unique] == 0,
                                  dens[global_unique],
                                  1.0e-8)
                del_pos = int(np.argmax(metric))
                del_sim_num = int(global_unique[del_pos])

                deleted = np.where(
                        (global_unique != del_sim_num) & (deleted[global_unique] != 1),
                        0, 1
                        ).astype(np.float64)
            keep = np.array([global_unique[i] for i in range(len(deleted)) if deleted[i] == 0])
        else:
            keep = metric(**kwargs)
        return keep

    def get_umap(self, **kwargs):
        """
        Initializes UMAP for the set features

        Args:
            **kwargs: Any UMAP arguments. pyRANN defaults use UMAP defaults
                except for the following: verbose=True, init="pca", n_neighbors=54.
                If kwargs are defined, all undefined UMAP args will default back
                to UMAP default.

        Returns:
            An instance of UMAP to visualize the descriptor space

        Raises:
            ValueError: If the features are not set
        """
        import umap
        # import umap.plot
        # if self.features == np.zeros((100,100)):
        if not np.any(self.features):
            raise ValueError('Please set the features before calling this function.')

        if not kwargs:
            mapper = umap.UMAP(verbose=True, init='pca', n_neighbors=54).fit(self.features)
        else:
            mapper = umap.UMAP(**kwargs)
        return mapper

    def plot(self,
             mapper,
             color_data: npt.ArrayLike,
             hover_info: dict,
             color_type: Optional[str] = 'categorical',
             subset_points: Optional[Union[npt.ArrayLike, None]] = None,
             key_str: Optional[Union[str, None]] = None):
        """
        Saves an interactive bokeh html plot to visualize the feature-space.

        Args:
            mapper: UMAP instance
            color_data: The data that determines how each data point in the 
                interactive plot is colored. 
            hover_info: Dictionary containing the information that will appear when a
                data point is hovered over in the interactive plot. The key will be
                the label and the values should be arrays of length natoms.
            color_type: Type of coloring to use. If categorical, each unique item
                in color_data gets its own color. If linear, viridis colormap will be
                used to color the data from min(color_data) to max(color_data).
                Defaults to categorical
            subset_points: 1-D array of booleans of length natoms. True means that 
                the data point for that atom should be plotted.
            key_str: The string to name the html file.
        """
        hover_data = pd.DataFrame(hover_info)
        data = pd.DataFrame(mapper.embedding_, columns=('x', 'y'))
        tooltip_dict = {}
        for col_name in hover_data:
            data[col_name] = hover_data[col_name]
            tooltip_dict[col_name] = '@{'+col_name+'}'
        tooltips = list(tooltip_dict.items())
        if color_type.lower() == 'categorical':
            unique, index = np.unique(color_data, return_index=True)
            index = np.argsort(index)
            unique = unique[index]
            color_key = [plc.to_hex(c) for c in plt.get_cmap('cet_glasbey_category10')(np.linspace(0,1,np.unique(color_data).shape[0]))]
            new_color = dict(zip(unique, color_key))
            data['label'] = color_data
            data['color'] = pd.Series(color_data).map(new_color)

            if subset_points:
                data['subset'] = subset_points
                data = data[data['subset']==True]
                data = data.drop('subset', axis=1)
                data = data.reset_index(drop=True)

            dfs = [data[data['label']==i] for i in unique]
            fig = bpl.figure(width=2000, height=1000, tooltips=tooltips,
                             tools=('pan,wheel_zoom,box_zoom,save,reset,help'),
                             background_fill_color='black')
            for i in range(len(dfs)):
                data_source = bpl.ColumnDataSource(dfs[i])
                fig.circle(x='x', y='y', source=data_source, color='color', size=2, legend_label=unique[i])
            fig.legend.location = 'top_left'
            fig.legend.click_policy = 'hide'
            fig.grid.visible=False
            fig.axis.visible=False
            if key_str:
                if key_str.endswith('.html'):
                    key_str = key_str.split('.')[0]
                bpl.output_file(filename=f'{key_str}.html')
            bpl.save(fig)
            # bpl.show(fig)
            # if key_str:
            #     py_file = os.path.basename(__file__)
            #     base_file = py_file.split('.')[0]
            #     os.system(f'mv {base_file}.html {key_str}.html')

        elif color_type.lower() == 'linear':
            data['color'] = color_data
            if subset_points:
                data['subset'] = subset_points
                data = data[data['subset']==True]
                data = data.drop('subset', axis=1)
                data = data.reset_index(drop=True)
            data_source = bpl.ColumnDataSource(data)
            visible = bpl.ColumnDataSource(data)
            color_mapper = LinearColorMapper(palette=Viridis256,
                                             low=min(color_data),
                                             high=max(color_data))
            fig = bpl.figure(width=2000, height=1000, tooltips=tooltips,
                             tools=('pan,wheel_zoom, box_zoom,save,reset,help'),
                             background_fill_color='black')
            fig.circle(x='x', y='y', source=visible,
                       color=transform('color', color_mapper),
                       size=2)
            color_bar = ColorBar(color_mapper=color_mapper, label_standoff=12)
            fig.add_layout(color_bar, 'left')
            if key_str:
                if key_str.endswith('.html'):
                    key_str = key_str.split('.')[0]
                bpl.output_file(filename=f'{key_str}.html')
            fig.grid.visible=False
            fig.axis.visible=False

            slider = RangeSlider(
                    start=min(color_data),
                    end=max(color_data),
                    value=(min(color_data), max(color_data)),
                    step=(max(color_data)-min(color_data))/200,
                    title='Value Range',
                    width=650,
                    )
            callback = CustomJS(
                    args=dict(source=data_source,
                              visible=visible,
                              slider=slider,
                              color_mapper=color_mapper),
                    code="""
                    const low = slider.value[0];
                    const high = slider.value[1];

                    const src_x = source.data['x'];
                    const src_y = source.data['y'];
                    const src_val = source.data['color'];

                    const new_x = [];
                    const new_y = [];
                    const new_val = [];

                    for (let i =0; i < src_val.length; i++) {
                        if (src_val[i] >= low && src_val[i] <= high) {
                            new_x.push(src_x[i]);
                            new_y.push(src_y[i]);
                            new_val.push(src_val[i]);
                        }
                    }

                    visible.data = { x: new_x, y: new_y, color: new_val };

                    color_mapper.low = low;
                    color_mapper.high = high;

                    visible.change.emit();
                    color_mapperchange.emit();
                """,
                )
            slider.js_on_change("value", callback)
            layout = column(fig, slider)

            bpl.save(layout)
            # TODO - Fix hovertools
            # bpl.show()
            # if key_str:
            #     py_file = os.path.basename(__file__)
            #     base_file = py_file.split('.')[0]
            #     os.system(f'mv {base_file}.html {key_str}.html')
        else:
            raise ValueError('color_type must either be "categorical" or "linear"')
