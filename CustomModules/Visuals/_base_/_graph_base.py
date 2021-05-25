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

from Visuals._base_._axes_base import AxesInstance
from Visuals._utils_._exceptions import *
from Visuals._utils_._type_definitions import *


class GraphBase:
           
    """ Class Static Methods """
    
    @staticmethod 
    def to_numpy(array: NumericArray) -> np.ndarray:
        if not isinstance(array, np.ndarray):
            # Testo o tipo da variável usando type() para tipos nativos do python por ser bastante legível
            # Se for explicitamente uma lista nativa do python...
            if type(array) is list:
                array = np.array(array)
            # Porém, para testar possíveis instâncias, sempre utilizar isinstance()
            elif isinstance(array, pd.DataFrame):
                array = array.values

            # Conforme bugs forem surgindo, atualizarei esse bloco de código
            # Para suportar novos tipos
            else:
                raise ValueError('Tipo de input não reconhecido')
        return array
    
    
    @staticmethod
    def numpy_convert(
            array: NumericArray,
            check: bool = True,
            squeeze: bool = False, 
            expand: bool = False,
            flat: bool = False
        ) -> np.ndarray:
        
        if check:
            ndarray = GraphBase.to_numpy(array)
        
        if squeeze and ndarray.ndim >= 2:
            ndarray = np.squeeze(ndarray)
            
        if expand and ndarray.ndim == 1:
            ndarray = np.expand_dims(ndarray, axis=0)
        
        if flat and ndarray.ndim >= 2:
                flatten = [feature.ravel() for feature in ndarray]
                ndarray = np.array(flatten).T
            
        return ndarray
    
    
    """ Plot assist Methods """
    

    def new_plot(
            self, 
            n_axis: int = None, 
            current_Axes: _subplots.Axes=None,
            labels: Iterable[str] = None, 
            fontsize: int = None, 
            lims: Tuples = None, 
            elev: int = None, 
            azim: int = None,
            grid: bool = False,
            title: str = None,
            **kwargs
        ) -> NoReturn:
        
        if current_Axes is not None:
            if not isinstance(current_Axes, _subplots.Axes):
                raise Error('InvalidAxesObject')    
            self.axObj = current_Axes
            
        elif n_axis is None:
            # se não for passado, o padrão é um plot 2D
            self.axObj = AxesInstance(n_axis=2, **kwargs)
        else:
            self.axObj = AxesInstance(n_axis, **kwargs)
        
        if labels is not None:
            self.axObj.set_ax_labels(labels, fontsize)
            
        if lims is not None:
            self.axObj.set_ax_limits(lims)
        
        if self.axObj.n_axis == 3:
            self.axObj.set_ax_view(elev, azim)
        
        if title is not None:
            self.axObj.set_ax_title(title)
        
        if grid:
            self.axObj.ax.grid()            
            
        
    
    def enable_legend(self) -> NoReturn:
        self.axObj.ax.legend()
        

    def full_coordinates(self, X: NumpyArray) -> bool:
        # Se X é unidimensional, então não pode conter coordenadas de multiplos eixos
        if X.ndim == 1:
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
                
    
    '''
    
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

    '''
    
    
    def iter_params(
            self,
            X: NumericArray, 
            Y: NumericArray = None, 
            Z: NumericArray = None,
        ) -> Iterator[NumericArray]:
        
        # Se Y e Z não foram passados, avalia X
        if all(_input_ is None for _input_ in [Y,Z]):
            print('ALL NONE')
            
            X = GraphBase.to_numpy(X)
            
            if X.ndim == 1:
                X = GraphBase.numpy_convert(X, expand=True)
            
            n_samples, n_features = X.shape
                
            if n_features == self.axObj.n_axis:
                print('Full coords')
                for coord in X.T:
                    yield coord
            
            else:
                raise ValueError('Input Não reconhecido')    
                    
        elif self.axObj.n_axis == 2:
            for coord in (X, Y):
                yield coord
                
        elif self.axObj.n_axis == 3 and Z is None:
            raise Error('MissingZ')
            
        else:
            for coord in (X, Y, Z):
                yield coord                    
                    
        
        
        
        
        
        
        

