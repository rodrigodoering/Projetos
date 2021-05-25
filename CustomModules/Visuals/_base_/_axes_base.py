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

from Visuals._utils_._exceptions import *
from Visuals._utils_._type_definitions import *


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
            raise Error('AxesInstance')
        
        self.n_axis = n_axis
        self.axis_ids = ['x','y'] if n_axis == 2 else ['x', 'y', 'z']
        self.last_function_call = None
       
    
    def get_new_axes(self, n_axis: int, **kwargs) -> NoReturn:
        if n_axis == 2:
            self.fig, self.ax = plt.subplots(**kwargs)   
        elif n_axis == 3:
            self.fig = plt.figure(**kwargs)
            self.ax = self.fig.gca(projection='3d')
        else:
            raise Error('NumberAxis')
     
        
    def get_ax_method(self, method_list: list) -> Generator[Callable, None, NoReturn]:
        """ Retorna funções de label """
        for func in method_list[:self.n_axis]:
            yield getattr(self.ax, func)


    def assert_sequence(self, _input_: Sequence, required: int = None) -> Sequence:
        if required is None:
            required = self.n_axis
        #print(_input_)
        size_input = len(_input_)
        assert_msg = 'Valores na Sequência: %d ; Exigido pela função: %d' 
        assert size_input == required, assert_msg % (size_input, required)


    
    def set_ax_labels(self, labels: list = None, fontsize: int = None) -> NoReturn:
        if labels is None:
            axis_labels = ('$x_%d$' % (i + 1) for i in range(self.n_axis))
        else:
            axis_labels = (_str_ for _str_ in labels)
            
        label_methods = ['set_%slabel' % id_ for id_ in self.axis_ids]
        fontsize = 16 if fontsize is None else fontsize
        print(label_methods)
        for method in self.get_ax_method(label_methods):
            method(next(axis_labels), fontsize=fontsize, rotation=0)  
                    
    
    def set_ax_limits(self, lims: Tuples = None) -> NoReturn:
        if lims is None:
            # se nenhum valor for passado
            pass
        else:
            
            if isinstance(lims, list):
                # se for uma lista de tuplas, avalia se existe uma para cada eixo
                self.assert_sequence(lims, self.n_axis)
                axis_lims = (_tuple_ for _tuple_ in lims)
                
            elif isinstance(lims, tuple):
                # Se for uma tupla apenas, os valores são usados para todos os eixos
                axis_lims = (lims for i in range(self.n_axis))
            
            else:
                raise TypeError('data type do input não previsto')
                
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
                    
            
    
    def control_plot_params(input_type: str) -> Callable: 
        # Função permite aplicar argumentos no decorador
        def decorator(func): 
            # função decoradora
            @functools.wraps(func)
            def wrapper(self, *args, **kwargs):
                # Valida a quantidade de inputs versus a quantidade de axis do plot
                # O tipo de plot + o número de eixos requerem quantidades específicas de coordenadas
                if input_type == 'coordinates':
                    required_vals = self.n_axis
                
                elif input_type == 'vector':
                    required_vals = self.n_axis 
                
                elif input_type == 'grid':
                    required_vals = 3
                    
                else:
                    raise Exception('Something weird happened')
                
                #print('CONTROL_PLOT_PARAMS')
                #print(args)
                
                self.last_function_call = func.__name__
                # DEBUG
                print('Call:', self.last_function_call)
                print('Len Args', len(args))
                print('required', required_vals, '\n')
                
                self.assert_sequence(args, required_vals)

                # retorna função do objeto ax
                return func(self, *args, **kwargs) 
            # retorna o decorador
            return wrapper  
        return decorator
          
    
    @control_plot_params(input_type='coordinates')
    def ax_text(self, *coords: NumericArray, text: str, **kwargs) -> NoReturn:
        self.ax.text(*coords, s=text, **kwargs)
    
    
    @control_plot_params(input_type='coordinates')
    def ax_plot(self, *coords: NumericArray,**kwargs) -> NoReturn:
        self.ax.plot(*coords, **kwargs)


    @control_plot_params(input_type='coordinates')
    def ax_scatter(self, *coords: NumericArray, **kwargs) -> NoReturn:
        self.ax.scatter(*coords, **kwargs)  
        
        
    @control_plot_params(input_type='vector')
    def ax_quiver(self, *coords: NumericArray, origin: tuple = None, **kwargs) -> NoReturn:      
        tail_coords = tuple(0 for i in range(self.n_axis)) if origin is None else origin
        self.ax.quiver(*tail_coords, *coords, **kwargs)
 

    @control_plot_params(input_type='grid')
    def ax_contourf(self, *coords: NumericArray, levels: int = None, **kwargs) -> NoReturn:
        self.ax.contourf(*coords, levels=levels, **kwargs)


    @control_plot_params(input_type='grid')
    def ax_surface(self, *coords, **kwargs) -> NoReturn:
        self.ax.plot_surface(*coords, **kwargs)