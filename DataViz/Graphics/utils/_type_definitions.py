# -*- coding: utf-8 -*-
"""
Created on Wed May 12 19:56:10 2021

@author: rodri
"""
import numpy as np

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
from typing import NewType


# Agrupa todos os tipos de dados em novos objetos Typing
# Facilita a documentação visto que a maioria das funções criadas suportam
# uma certa flexibilidade de tipos de dados nos inputs


Numeric = NewType(
    'Numeric', 
    Union[int, float, complex]
)


NumericArray = NewType(
    'NumericArray',
    Union[List[Numeric], np.ndarray, np.matrix]
)


SequenceLenght = NewType(
    'SequenceLenght',
    Union[int, Sequence[Any]]
)


NumpyArray = NewType(
    'NumpyObject',
    Union[np.ndarray, np.matrix]
)

Strings = NewType(
    'Strings',
    Union[str, List[str]]
)







