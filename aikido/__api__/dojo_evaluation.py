import logging
from dataclasses import dataclass
from typing import TypeVar, Generic, List

import numpy as np

T = TypeVar('T')

logger = logging.getLogger(__name__)


@dataclass
class DojoEvaluation(Generic[T]):
    pred: np.ndarray
    loss: np.ndarray
    metrics: List[dict]
