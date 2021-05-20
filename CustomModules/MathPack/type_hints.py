import numpy as np
import pandas as pd

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
    Union[List[Numeric], np.ndarray, np.matrix, pd.core.DataFrame]
)


Sequence = NewType(
    'SequenceLenght',
    Union[Sequence[Any], tuple, list, np.ndarray]
)


NumpyArray = NewType(
    'NumpyObject',
    Union[np.ndarray, np.matrix]
)

Strings = NewType(
    'Strings',
    Union[str, List[str]]
)


Tuples = NewType(
    'Tuples',
    Union[Iterable[tuple], tuple]
)
