# -*- coding: utf-8 -*-
"""
Created on Wed May 12 19:56:10 2021

@author: rodri
"""


class Error(Exception):
    
    def __init__(self, _error_: str):
           
        Erros = {
            
            'AxesInstance':'Parâmetro "current_Axes": objeto matplotlib.axes._subplots.Axes',
            
            'NumberAxis': 'Parâmetro n_axis deve ser 2 (plot bidimensional) ou 3 (plot tridimensional)',
            
            'AxisCoordinates': 'O número de coordenadas passadas é incompatível com número de eixos',
            
            'AxisFunction': 'Essa função não se aplica ao número atual de eixos do plot',
            
            'MissingZ':'O plot é tridimensional, e os valores da terceira dimensão não foram passados',
            
            'InsufficientInput': """Inputs insuficientes para gerar o plot: \nPasse um grid de coordenadas pronto ou então uma função Callable"""

        }
        
        if _error_ not in Erros.keys():
            error_msg = 'Ocorreu um erro não especificado'
        
        else:
            error_msg = Erros[_error_]
            
        super(Error, self).__init__(error_msg)
    
    

