# -*- coding: utf-8 -*-
"""
Created on Sun Apr  4 11:03:02 2021

@author: rodrigo doering neves
@email: rodrigodoeringneves@gmail.com
"""

import pandas as pd
import numpy as np
import seaborn as sb
import functools

import matplotlib.pyplot as plt
from matplotlib.axes import _subplots

# User defined errors
import _exceptions
from _axes_base import AxesInstance

from typing import NoReturn
from typing import Optional
from typing import Iterator 
from typing import Callable
from typing import Generator 
from typing import Iterable
from typing import Union
from typing import List
from typing import Any
from typing import Sequence

Numeric = Union[int, float, complex]
NumericArray = Union[List[Numeric], np.ndarray, np.matrix]
NumpyArray = Union[np.ndarray, np.matrix]
SequenceLenght = Union[int, Sequence[Any]]


class GraphBase:
    
    def __init__(self, n_axis: int, current_Axes: _subplots.Axes=None, **kwargs):
        self.axObj = AxesInstance(n_axis=n_axis, current_Axes=current_Axes, **kwargs)
        

    def new_plot(self, n_axis: int = None, **kwargs) -> NoReturn:
        if n_axis is None:
            self.axObj = AxesInstance(n_axis=self.axObj.n_axis, **kwargs)
        else:
            self.axObj = AxesInstance(n_axis, **kwargs)

    
    @staticmethod
    def numpy_convert(array: NumericArray) -> np.ndarray:
        if isinstance(array, list):
            array = np.array(array)
        return array


    @staticmethod
    def built_spec_kwargs(args: dict = None) -> dict:
        specs_keys = ['labels', 'fontsize', 'lims', 'elev', 'azim', 'title', 'grid', 'legend']
        specs = {key:None for key in specs_keys}
        for key in ['grid', 'legend']:
            specs[key] = False
        if args is not None:
            for key in args.keys():
                specs[key] = args[key]
        return specs
    
                
    def iter_params(
            self,
            X: NumericArray, 
            Y: NumericArray, 
            Z: NumericArray = None,
        ) -> Iterator[NumericArray]:
        if self.axObj.n_axis == 2:
            for coord in (X, Y):
                yield coord
                
        elif self.axObj.n_axis == 3 and Z is None:
            raise _exceptions.MissingZError
            
        else:
            for coord in (X, Y, Z):
                yield coord


    def annotate(
            self, 
            coords: NumericArray, 
            annotations: Iterable[str], 
            offset: float = 0.1, 
            ax_offset: int = 0, 
            **kwargs
        ) -> NoReturn:
        
        coords = GraphBase.numpy_convert(coords)
        
        # Se um axis inexistente for passado, zera o offset
        if ax_offset > self.axObj.n_axis:
            print('AxesInstance: ax_offset não existe no plot, desconsiderando offset')
            offset = 0
        
        for vec, _str_ in zip(coords, annotations):
            coord_vals = (vec[i] + offset * int(i == ax_offset) for i in range(self.axObj.n_axis))
            self.axObj.ax_text(*coord_vals, text=_str_, **kwargs)


    def set_plot_specs(
            self, 
            labels: Iterable[str] = None, 
            fontsize: int = None, 
            lims: Iterable[tuple] = None, 
            elev: int = None, 
            azim: int = None,
            grid: bool = None,
            title: str = None,
            legend: bool = False
        ) -> NoReturn:
        
        self.axObj.set_ax_labels(labels, fontsize)
        self.axObj.set_ax_limits(lims)
        
        if self.axObj.n_axis == 3:
            self.axObj.set_ax_view(elev, azim)
        
        if title is not None:
            self.axObj.set_ax_title(title)
        
        if grid:
            self.axObj.ax.grid()
        
        if legend:
            self.axObj.ax.legend()


    def generate_grid_coords(
            self, 
            min_val: Numeric = -1, 
            max_val: Numeric = 1, 
            n_vals: int = 5, 
            dim_func: Callable = None, 
            return_flat: bool = False
        ) -> NumericArray:
        
        base_var = np.linspace(min_val, max_val, n_vals)
        update_axis = lambda i: (slice(np.newaxis),) + (np.newaxis,) * i
        # Cria o grid de valores, len(grid) = n_dims
        grid = np.broadcast_arrays(*(base_var[update_axis(i)] for i in range(self.axObj.n_axis)))
        
        # Se for passada uma função para a última dimensão
        if dim_func is not None:
            dim_grid_shape = tuple(n_vals for i in range(self.axObj.n_axis))
            last_dim_grid = dim_func(np.array([vec.flatten() for vec in grid[:-1]]).T).reshape(dim_grid_shape)
            grid[-1] = last_dim_grid
            
        if return_flat:
            # retorna os valores do grid como rowvectors de shape (n_vals^n_dims x n_dims)
            return np.array([vec.flatten() for vec in grid]).T
    
        return grid