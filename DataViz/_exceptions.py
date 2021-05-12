# -*- coding: utf-8 -*-
"""
Created on Wed May 12 19:56:10 2021

@author: rodri
"""


class UserDefinedError(Exception):
    """Classe base para definir os erros"""
    pass
    

class AxesInstanceError(UserDefinedError):
    """Parâmetro "current_Axes": objeto matplotlib.axes._subplots.Axes"""
    pass

class NumberAxisError(UserDefinedError):
    """Parâmetro n_axis deve ser 2 (plot bidimensional) ou 3 (plot tridimensional)"""
    pass

class AxisCoordinatesError(UserDefinedError):
    """ O número de coordenadas passadas é incompatível com número de eixos"""
    pass
    

class AxisFunctionError(UserDefinedError):
    """ Essa função não se aplica ao número atual de eixos do plot"""
    pass


class MissingZError(UserDefinedError):
    """O plot é tridimensional, e os valores da terceira dimensão não foram passados"""
    pass
