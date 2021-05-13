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
SequenceLenght = Union[int, Sequence[Any]]


class AxesInstance:
    
    """ Representa o Objeto Axes """
    
    
    def __init__(self, n_axis: int, current_Axes: _subplots.Axes=None, **kwargs) -> NoReturn:
        
        if current_Axes is None:
            self.get_new_axes(n_axis, **kwargs)
                
        elif isinstance(current_Axes, _subplots.Axes):
            n_axis = 3 if 'Axes3DSubplot' in current_Axes.__str__() else 2
            self.fig = plt.figure
            self.ax = current_Axes
            
        else:
            raise _exceptions.AxesInstanceError
        
        self.n_axis = n_axis
        self.axis_ids = ['x','y'] if n_axis == 2 else ['x', 'y', 'z']       
       
    
    def get_new_axes(self, n_axis: int, **kwargs) -> NoReturn:
        if n_axis == 2:
            self.fig, self.ax = plt.subplots(**kwargs)   
        elif n_axis == 3:
            self.fig = plt.figure(**kwargs)
            self.ax = self.fig.gca(projection='3d')
        else:
            raise _exceptions.NumberAxisError
     
        
    def get_ax_method(self, method_list: list) -> Generator[Callable, str, NoReturn]:
        """ Retorna funções de label """
        for func in method_list[:self.n_axis]:
            yield getattr(self.ax, func)
       
    
    def set_ax_labels(self, labels: list = None, fontsize: int = None) -> NoReturn:
        
        if labels is None:
            axis_labels = ('$x_%d$' % (i + 1) for i in range(self.n_axis))
        else:
            axis_labels = (_str_ for _str_ in labels)
            
        label_methods = ['set_%slabel' % id_ for id_ in self.axis_ids]
        fontsize = 16 if fontsize is None else fontsize
        
        for method in self.get_ax_method(label_methods):
            method(next(axis_labels), fontsize=fontsize, rotation=0)  
                    
    
    def set_ax_limits(self, lims: Iterable[tuple] = None) -> NoReturn:
        if lims is None:
            axis_lims = ((None,None) for i in range(self.n_axis))
        else:
            axis_lims = (_tuple_ for _tuple_ in lims)
        lim_methods = ['set_%slim' % id_ for id_ in self.axis_ids]
        for method in self.get_ax_method(lim_methods):
            method(next(axis_lims))
                
                
    def set_ax_view(self, elev: int = None, azim: int = None) -> NoReturn:
        if self.n_axis != 3:
            raise _exceptions.AxisFunctionError
        self.ax.view_init(elev=elev, azim=azim)
    
        
    def set_ax_title(self, title: str = None) -> NoReturn:
        if title is None:
            pass
        else:
            self.ax.set_title(title)
                    
    
    def validate_sequence(self, _input_: SequenceLenght, required: int):
        size_input = _input_ if isinstance(_input_, int) else len(_input_)
        if size_input != required:
            raise _exceptions.AxisCoordinatesError
    
    
    def control_plot_params(plot_type: str) -> Callable: 
        
        def decorator(func): 
            
            @functools.wraps(func)
            def wrapper(self, *args, **kwargs):
                if plot_type == 'coordinates':
                    required_vals = self.n_axis
                
                elif plot_type == 'vector':
                    required_vals = self.n_axis * 2
                
                elif plot_type == 'surface':
                    required_vals = 3
                    
                else:
                    raise NotImplementedError
                    
                # DEBUG
                print('Call:', func.__name__)
                print('Len Args', len(args))
                print('required', required_vals, '\n')
                #print(args)
                self.validate_sequence(len(args), required_vals)
                                
                return func(self, *args, **kwargs) 
            
            return wrapper  
        
        return decorator
      
    
    @control_plot_params(plot_type='coordinates')
    def ax_text(self, *coords: NumericArray, text: str, **kwargs) -> NoReturn:
        self.ax.text(*coords, s=text, **kwargs)
    
    
    @control_plot_params(plot_type='coordinates')
    def ax_plot(self, *coords: NumericArray,**kwargs) -> NoReturn:
        self.ax.plot(*coords, **kwargs)


    @control_plot_params(plot_type='coordinates')
    def ax_scatter(self, *coords: NumericArray, **kwargs) -> NoReturn:
        self.ax.scatter(*coords, **kwargs)  
        
        
    @control_plot_params(plot_type='vector')
    def ax_quiver(self, *coords: NumericArray, origin: tuple = None, **kwargs) -> NoReturn:      
        tail_coords = tuple(0 for i in range(self.n_axis)) if origin is None else origin
        self.ax.quiver(*tail_coords, *coords, **kwargs)
 

    @control_plot_params(plot_type='surface')
    def ax_contourf(self, *coords: NumericArray, levels: int = None, **kwargs) -> NoReturn:
        self.ax.contourf(X, Y, levels=levels, **kwargs)


    @control_plot_params(plot_type='surface')
    def ax_surface(self, *coords, **kwargs) -> NoReturn:
        raise NotImplementedError