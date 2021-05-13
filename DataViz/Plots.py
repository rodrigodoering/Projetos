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
from _graph_base import GraphBase

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


class Plots(GraphBase):
    
    """ Representa o gráfico, herda de AxesInstance """
       
    
    def __init__(self, n_axis: int, current_Axes: _subplots.Axes=None, **kwargs) -> NoReturn:
        # Inicia os parâmetros 
        super(Plots, self).__init__(n_axis, current_Axes=current_Axes, **kwargs)
    
    
    def Function(
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
    

    def Scatter(
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
        
       self.axObj.ax_scatter(*self.iter_params(X, Y, Z), **kwargs)

       
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
   
    

          
    def Surface(self):
        pass
                  