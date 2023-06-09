from dtorch.typing import types, DtOptional, Optional

@types(ratio=DtOptional(int), tg = int, return_type=DtOptional(int))
def function_1(tg : int, ratio : Optional[int] = 4):
    print(ratio)
    return ratio

@types(ratio=float, return_type=None)
def function_2(ratio : float):
    print(ratio)

function_1(1, ratio=2)
function_2(2.3)
