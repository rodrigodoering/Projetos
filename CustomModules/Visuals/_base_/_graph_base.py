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

import sys
sys.path.append('C:\\Users\\rodri\\Desktop\\Relacionados a Código\\github_Rodrigo\\Notebooks\\DataViz_Obj')

import matplotlib.pyplot as plt
from matplotlib.axes import _subplots

from Visuals._base_._axes_base import AxesInstance
from Visuals._utils_._exceptions import *
from Visuals._utils_._type_definitions import *


class GraphBase:
           
    """ Class Static Methods """
    
    @staticmethod
    def numpy_convert(array: NumericArray) -> np.ndarray:
        if isinstance(array, list):
            array = np.array(array)
        return array


    @staticmethod
    def build_spec_kwargs(args: dict = None) -> dict:
        specs_keys = ['labels', 'fontsize', 'lims', 'elev', 'azim', 'title', 'grid']
        specs = {key:None for key in specs_keys}
        specs['grid'] = False
        
        if args is not None:
            for key in args.keys():
                specs[key] = args[key]
        return specs
    
    
    @staticmethod
    def flat_grid(grid):
        flatten = [coords.ravel() for coords in grid]
        return np.array(flatten).T
    
    
    """ Plot assist Methods """
    

    def new_axes_obj(self, n_axis: int = None, current_Axes: _subplots.Axes=None, **kwargs) -> NoReturn:
        if current_Axes is not None:
            if not isinstance(current_Axes, _subplots.Axes):
                raise Error('InvalidAxesObject')
            self.axObj = current_Axes
        elif n_axis is None:
            # se não for passado, o padrão é um plot 2D
            self.axObj = AxesInstance(n_axis=2, **kwargs)
        else:
            self.axObj = AxesInstance(n_axis, **kwargs)    
     
    
    def enable_legend(self) -> NoReturn:
        self.axObj.ax.legend()
        

    def set_plot_specs(
            self,
            labels: Iterable[str] = None, 
            fontsize: int = None, 
            lims: Tuples = None, 
            elev: int = None, 
            azim: int = None,
            grid: bool = None,
            title: str = None,
        ) -> NoReturn:
              
        self.axObj.set_ax_labels(labels, fontsize)
        self.axObj.set_ax_limits(lims)
        
        if self.axObj.n_axis == 3:
            self.axObj.set_ax_view(elev, azim)
        
        if title is not None:
            self.axObj.set_ax_title(title)
        
        if grid:
            self.axObj.ax.grid()
        
    
    
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
        # Possivelmente, essa lógica poderá ser revista mais a frente para suportar tensores
        else:
            return False
                
    
    
    def iter_params(
            self,
            X: NumericArray, 
            Y: NumericArray = None, 
            Z: NumericArray = None,
        ) -> Iterator[NumericArray]:
        
        is_none = (_input_ is None for _input_ in [Y,Z])
        
        if self.full_coordinates(X) and all(is_none):
            print('Full coords')
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



