""" imports """
from typing import Any

class Operation:

    def __call__(self, *args) -> Any:
        pass

    def __str__(self) -> str:
        pass

    def backward(self, base_tensor = None) -> Any:
        pass
