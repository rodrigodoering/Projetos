# -*- coding: utf-8 -*-
"""
Spyder Editor

Script para desenvolvimento.
"""
import pandas as pd
import numpy as np
import seaborn as sb

import matplotlib.pyplot as plt
from matplotlib.axes import _subplots
import _utils

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

####################################### Class: AxesInstance ######################################## 


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
            raise('Parâmetro "current_Axes": objeto matplotlib.axes._subplots.Axes')
        
        self.n_axis = n_axis
        self.axis_ids = ['x','y'] if n_axis == 2 else ['x', 'y', 'z']       
       
    
    def get_new_axes(self, n_axis: int, **kwargs) -> NoReturn:
        if n_axis == 2:
            self.fig, self.ax = plt.subplots(**kwargs)   
        elif n_axis == 3:
            self.fig = plt.figure(**kwargs)
            self.ax = self.fig.gca(projection='3d')
        else:
            raise('n_axis deve ser 2 ou 3')
        
        
    def get_ax_method(self, method_list: list) -> Generator[Callable, str, None]:
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
            raise('Supported only for 3D Plots')
        self.ax.view_init(elev=elev, azim=azim)
    
        
    def set_ax_title(self, title: str = None) -> NoReturn:
        if title is None:
            pass
        else:
            self.ax.set_title(title)
                    
    
    def validate_sequence(self, _input_: Sequence[Any]):
        if len(_input_) != self.n_axis:
            raise('Input incompatível com número de eixos do plot')
 
    
    def iter_params(
            self, 
            x: NumericArray, 
            y: NumericArray, 
            z: NumericArray = None
        ) -> Iterator[NumericArray]:
        if self.n_axis == 2:
            for coord in (x, y):
                yield coord
                
        elif self.n_axis == 3 and z is None:
            raise('O plot é tridimensional, e os valores de Z não foram passados')
            
        else:
            for coord in (x, y, z):
                yield coord
    
    
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
            self.ax.text(*coord_vals, s=_str_, **kwargs)
        
           
    def ax_plot(
            self, 
            x: NumericArray, 
            y: NumericArray, 
            z: NumericArray = None,
            **kwargs
        ) -> NoReturn:
        self.ax.plot(*self.iter_params(x, y, z), **kwargs)
    
    
    def ax_scatter(
            self, 
            x: NumericArray, 
            y: NumericArray, 
            z: NumericArray = None,
            **kwargs
        ) -> NoReturn:
        self.ax.scatter(*self.iter_params(x, y, z), **kwargs)
    
    
    def ax_quiver(
            self, 
            x: NumericArray, 
            y: NumericArray, 
            z: NumericArray = None,
            origin: tuple = None,
            **kwargs
        ) -> NoReturn:
        
        tail_coords = tuple(0 for i in range(self.n_axis)) if origin is None else origin
        head_coords = self.iter_params(x, y, z)
        ax.quiver(*tail_coords, *head_coords, **kwargs)
    

    def ax_contourf(
            self,
            X: NumericArray,
            Y: NumericArray,
            Z: NumericArray,
            levels: int = None,
            **kwargs
        ) -> NoReturn:
        self.ax.contourf(X, Y, levels=levels, **kwargs)

    
    def ax_surface(
            self,
            X,
            Y,
            Z,
            **kwargs
        ) -> NoReturn:
        raise NotImplementedError
        
        
    
    

########################################## Class: Graph ############################################        
        
        
class Graph:
    
    """ Representa o gráfico, herda de AxesInstance """
       
    
    def __init__(self, n_axis: int, current_Axes: _subplots.Axes=None, **kwargs) -> NoReturn:
        # Inicia os parâmetros 
        self.plot = AxesInstance(n_axis=n_axis, current_Axes=current_Axes, **kwargs)
        self.plot_dims = n_axis
    
    
    def new_plot(self, n_axis: int = None, **kwargs) -> NoReturn:
        if n_axis is None:
            self.plot.get_new_axes(self.plot_dims, **kwargs)
        else:
            self.plot.get_new_axes(n_axis, **kwargs)
            self.plot_dims = n_axis
        
    
    def built_spec_kwargs(self, args: dict = None) -> dict:
        specs_keys = ['labels', 'fontsize', 'lims', 'elev', 'azim', 'title', 'grid', 'legend']
        specs = {key:None for key in specs_keys}
        for key in ['grid', 'legend']:
            specs[key] = False
        if args is not None:
            for key in args.keys():
                specs[key] = args[key]
        return specs

    
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
        
        self.plot.set_ax_labels(labels, fontsize)
        self.plot.set_ax_limits(lims)
        
        if self.plot_dims == 3:
            self.plot.set_ax_view(elev, azim)
        
        if title is not None:
            self.plot.set_ax_title(title)
        
        if grid:
            self.plot.ax.grid()
        
        if legend:
            self.plot.ax.legend()
    
    
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
        
        n_features = self.plot_dims - 1
        
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
        
        self.plot.ax_plot(*X.T, Y, label=function_label, **kwargs)
        
        if plot_intercept:
            zero_vec = np.zeros(n_features)
            intercept = function(zero_vec)
            p0 = (0 if i + 1 < self.plot_dims else intercept[0] for i in range(self.plot_dims))
            self.plot.ax_scatter(*p0, color='red', marker='X')
        
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
       self.plot.ax_scatter(X, Y, Z, **kwargs)
       
       if annot is not None:
           coords = np.stack([*self.plot.iter_params(X,Y,Z)], axis=1)
           self.plot.annotate(
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
   
    
   def contourf
          
           
                  
           

######################################## EXECUÇÃO DO SCRIPT ########################################

if __name__ == '__main__':
    
    ### CRIA DADOS PARA TESTAR ###
    from sklearn.datasets import make_classification
    
    X, Y = make_classification(
        n_samples=300, 
        n_features=2,
        n_redundant=0,
        class_sep=2.2,
        n_clusters_per_class=1,
        flip_y=0.00,
        random_state=11,
        n_classes=2
    )
    
    # Cria um numpy array contendo o vetor de coeficientes w
    w = np.array([ 8.89, 0.54])
    # Cria um escalar para representar o bias
    bias = -1

    def linearFunc(X):
        # Retorna Y
        return np.dot(X,-3) + 2
    
    S = np.array([[-3,2],
                  [0,2],
                  [3,1]])

    
    Z = np.dot(S,w) + bias
    
    annot_sample = ['$f\,(S_%d) = %.2f$' % (i,z) for i,z in enumerate(Z)]
    
   
    graph_params = {
        'title':'Gráfico de teste',
        'labels':None,
        'legend':True
        }
    
    plot = Graph(n_axis=2, figsize=(10,7))   
    
    plot.scatter(X[:,0], X[:,1], c=Y, cmap=plt.cm.coolwarm, alpha=.2)
    plot.scatter(S[:,0], S[:,1], annot=annot_sample, annot_ax_offset=1, marker='x', color='black')
    plot.function(
        linearFunc, 
        domain=(-4,4), 
        function_label='$f(x)$',
        plot_intercept=True,
        specs={
            'lims':[(-4,4),(-1,5)]
        }
    )

    
    '''
    plot.plot_function(
        np.sin, 
        domain=(-2,2), 
        function_label='$sin$', 
        specs=graph_params, 
        n_samples=30, 
        color='green'
    )
    
     
    plot.plot_function(
        np.cos, 
        domain=(-2,2), 
        function_label='$cos$', 
        specs=graph_params, 
        n_samples=30, 
        color='red'
    )
    
    '''
    

    


    

      
      
      