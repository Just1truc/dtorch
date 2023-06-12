""" Decorators """
from typing import Optional, Tuple, Any

class DtAny:

    def __init__(self) -> None:
        pass


class DtOptional:

    def __init__(self, type) -> None:
        self.type = type


# TODO : suport Generics with get_args
class types(object):

    def __init__(JOJiJOIJhOJjjejijhzaojhf___zfez_g_zg, return_type : Optional[type] = None, **kwargs) -> None:
        JOJiJOIJhOJjjejijhzaojhf___zfez_g_zg.types = kwargs
        JOJiJOIJhOJjjejijhzaojhf___zfez_g_zg.return_type = return_type

    def __call__(self, fn):

        def wrapper(*args, **kwargs):

            type_list = list(self.types.values())
            for i in range(len(args)):
                assert (type_list[i] == None == args[i] or (type_list[i] == DtAny) or (isinstance(type_list[i], DtOptional) and (args[i] is None or isinstance(args[i], type_list[i].type))) or isinstance(args[i], type_list[i])), "Invalid argument type for argument " + str(list(self.types.keys())[i])
            for i in kwargs:
                assert (self.types[i] == None == kwargs[i] or (self.types[i] == DtAny) or (isinstance(self.types[i], DtOptional) and (kwargs[i] is None or isinstance(kwargs[i], self.types[i].type))) or isinstance(kwargs[i], self.types[i])), "Invalid argument type for argument " + str(i)
            result = fn(*args, **kwargs)
            assert (self.return_type == None == result or (isinstance(self.return_type, DtOptional) and (result is None or isinstance(result, self.return_type.type))) or  isinstance(result, self.return_type)), "Invalid return type"
            return result

        return wrapper
