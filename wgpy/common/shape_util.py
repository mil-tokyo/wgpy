from typing import List, Tuple, Union


def calculate_c_contiguous_strides(shape: Tuple[int, ...], itemsize: int) -> Tuple[int, ...]:
    new_strides = []
    stride = itemsize
    for s in reversed(shape):
        new_strides.insert(0, stride)
        stride *= s
    return tuple(new_strides)


def args_to_tuple_of_int(args: List[Union[int, Tuple[int, ...]]]):
    if len(args) == 1:
        try:
            return tuple(args[0]) # if args[0] is iterable
        except:
            pass
    return tuple(args)


def axis_to_tuple(axis: Union[int, Tuple[int]], sort=False) -> Tuple[int]:
    if isinstance(axis, int):
        axis = (axis, )
    if sort:
        axis = tuple(sorted(axis))
    return axis
