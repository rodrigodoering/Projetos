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


from .plots._axes_base import AxesInstance
from ..utils._exceptions import *
from ..utils._type_definitions import *


class GraphBase:
    
    
    def __init__(self, n_axis: int, current_Axes: _subplots.Axes=None, **kwargs):
        self.axObj = AxesInstance(n_axis=n_axis, current_Axes=current_Axes, **kwargs)
        self.customized_specs = False
        self.last_call = None
        

    def new_plot(self, n_axis: int = None, **kwargs) -> NoReturn:
        if n_axis is None:
            self.axObj = AxesInstance(n_axis=self.axObj.n_axis, **kwargs)
        else:
            self.axObj = AxesInstance(n_axis, **kwargs)
        self.customized_specs = False

    
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
    
    
    @staticmethod
    def flat_grid(grid):
        flatten = [coords.ravel() for coords in grid]
        return np.array(flatten).T

    

    def set_plot_specs(
            self,
            function_id: str,
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
        
        self.customized_specs = True                
        self.last_call = function_id
    
    
    
    def control_plot_specs(self, specs, function_name):
        if specs is not None and not self.customized_specs:
            plot_specs = GraphBase.built_spec_kwargs(specs)
            # function_id é um argumento posicional de set_plot_specs(), 
            # declarei o argumento para legibilidade
            self.set_plot_specs(function_id=function_name, **plot_specs)
        
        elif specs is not None and self.customized_specs:
            print('Specs já definido durante a chamada da função %s' % self.last_call)
        
        else:
            pass
    
    
    def full_coordinates(self, X: NumericArray) -> bool:
        X_numpy = GraphBase.numpy_convert(X)
        
        # Se X é unidimensional, então não pode conter coordenadas de multiplos eixos
        if X_numpy.ndim == 1:
            return False
        
        # Se as condições acimas não foram satisfeitas, as dimensões de X serão testadas
        elif X.ndim == 2:
            # Assume-se também que X está orientado como samples x features
            n_samples, n_features = X.shape
            
            if n_features == self.axObj.n_axis:
                return True
            
            else:
               return False 
            
        # Para tensores (ndim > 2), mantem-se os inputs como foram passados
        else:
            return False
                
    
    
    def iter_params(
            self,
            X: NumericArray, 
            Y: NumericArray, 
            Z: NumericArray = None,
        ) -> Iterator[NumericArray]:
        is_none = (_input_ is None for _input_ in [Y,Z])
        
        if self.full_coordinates(X) and all(is_none):
            for coord in X.T:
                yield coord
        
        elif self.axObj.n_axis == 2:
            for coord in (X, Y):
                yield coord
                
        elif self.axObj.n_axis == 3 and Z is None:
            raise Error('MissingZ')
            
        else:
            for coord in (X, Y, Z):
                yield coord



