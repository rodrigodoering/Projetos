# -*- coding: utf-8 -*-
"""
Created on Wed May 12 19:56:10 2021

@author: rodri
"""


   
class AxesInstanceError(Exception):
    
    def __init__(self):
        error_msg = 'Parâmetro "current_Axes": objeto matplotlib.axes._subplots.Axes'
        super(AxesInstanceError, self).__init__(error_msg)



class NumberAxisError(Exception):
    
    def __init__(self):
        error_msg = 'Parâmetro n_axis deve ser 2 (plot bidimensional) ou 3 (plot tridimensional)'
        super(NumberAxisError, self).__init__(error_msg)
   

class AxisCoordinatesError(Exception):
    
    def __init__(self):
        error_msg = 'O número de coordenadas passadas é incompatível com número de eixos'
        super(AxisCoordinatesError, self).__init__(error_msg)

    

class AxisFunctionError(Exception):
    
    def __init__(self):
        error_msg = 'Essa função não se aplica ao número atual de eixos do plot'
        super(AxisFunctionError, self).__init__(error_msg)



class MissingZError(Exception):
    
    def __init__(self):
        error_msg = 'O plot é tridimensional, e os valores da terceira dimensão não foram passados'
        super(MissingZError, self).__init__(error_msg)


