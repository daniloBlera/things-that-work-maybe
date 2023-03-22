#!/usr/bin/env python3
from datetime import datetime


def time_logger(func):
    def time_log_func(*args, **kwargs):
        time = datetime.now().strftime('%H:%M:%S')
        print(f'[{time}] {func.__qualname__}:{args} {kwargs}')
        return func(*args, **kwargs)

    return time_log_func


@time_logger
def my_func(x: int, y: int, z: int) -> int:
    return x + y + z


if __name__ == '__main__':
    # displaying args and kwargs
    print(f'{my_func(1, 2, 3)=}')
    print(f'{my_func(x=1, y=2, z=3)=}')
