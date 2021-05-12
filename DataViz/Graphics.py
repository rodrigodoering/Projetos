# -*- coding: utf-8 -*-
"""
Spyder Editor

Script para desenvolvimento.
"""
import pandas as pd
import numpy as np
import seaborn as sb
import functools

import matplotlib.pyplot as plt
from matplotlib.axes import _subplots

# User defined errors
import _exceptions

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


from _utils import Graph_Utils
from _axes_instance



class Graph(Graph_Utils):
    
    """ Representa o gráfico, herda de AxesInstance """
       
    
    def __init__(self, n_axis: int, current_Axes: _subplots.Axes=None, **kwargs) -> NoReturn:
        # Inicia os parâmetros 
        self.axObj = AxesInstance(n_axis=n_axis, current_Axes=current_Axes, **kwargs)
        super(Graph, self).__init__(n_axis)
    
    
    def new_plot(self, n_axis: int = None, **kwargs) -> NoReturn:
        if n_axis is None:
            self.plot = AxesInstance(n_axis=self.n_axis, **kwargs)
        else:
            self.axObj = AxesInstance(n_axis, **kwargs)

 
    def annotate(
            self, 
            coords: NumericArray, 
            annotations: Iterable[str], 
            offset: float = 0.1, 
            ax_offset: int = 0, 
            **kwargs
        ) -> NoReturn:
        
        coords = _utils.numpy_convert(coords)
        
        # Se um axis inexistente for passado, zera o offset
        if ax_offset > self.n_axis:
            print('AxesInstance: ax_offset não existe no plot, desconsiderando offset')
            offset = 0
        
        for vec, _str_ in zip(coords, annotations):
            coord_vals = (vec[i] + offset * int(i == ax_offset) for i in range(self.n_axis))
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
    
    
    def function(
            self, 
            function: Callable, 
            domain: Union[NumericArray, tuple] = None,
            n_samples: int = 10,
            plot_intercept: bool = False,
            specs: dict = None,
            function_label: str = '$f(x)$',
            **kwargs
        ) -> NoReturn:
        
        n_features = self.axObj.n_axis - 1
        
        if domain is None:
            X = np.array(
                [np.linspace(0,1,n_samples) for i in range(n_features)]
                ).reshape(n_samples, n_features)
    
        elif isinstance(domain, tuple):
            _min_ = domain[0]
            _max_ = domain[1]
            X = np.array(
                [np.linspace(_min_, _max_, n_samples) for i in range(n_features)]
                ).reshape(n_samples, n_features)           
        
        else:
            X = domain
        
        Y = function(X)
        
        self.axObj.ax_plot(*X.T, Y, label=function_label, **kwargs)
        
        if plot_intercept:
            zero_vec = np.zeros(n_features)
            intercept = function(zero_vec)
            p0 = (0 if i + 1 < self.axObj.n_axis else intercept[0] for i in range(self.axObj.n_axis))
            self.axObj.ax_scatter(*p0, color='red', marker='X')
        
        if specs is not None:
            plot_specs = self.built_spec_kwargs(specs)
            self.set_plot_specs(**plot_specs)
    

    def scatter(
            self,
            X: NumericArray, 
            Y: NumericArray, 
            Z: NumericArray = None,
            annot: Iterable[str] = None,
            annot_offset: float = 0.1,
            annot_ax_offset: int = 0,
            annot_fontsize: int = 12,
            annot_color: str = 'black',
            specs: dict = None,
            **kwargs
        ):
       self.axObj.ax_scatter(X, Y, Z, **kwargs)
       
       if annot is not None:
           coords = np.stack([*self.iter_params(X,Y,Z)], axis=1)
           self.annotate(
               coords = coords, 
               annotations = annot, 
               offset = annot_offset, 
               ax_offset = annot_ax_offset, 
               fontsize = annot_fontsize,
               color = annot_color
           )
       if specs is not None:
           plot_specs = self.built_spec_kwargs(specs)
           self.set_plot_specs(**plot_specs)
   
    
