"""In this script, we will wrap a Callable in a callable class with __call__ to use its attribute as a cache"""

import time
from typing import Callable


def expensive_function(any_input):
    """
    This function mocks the behaviour of a slow, deterministic function. Specifically, here the function is just
    an identity (of any possible input) with a time wait of 5 seconds.

    :param any_input: Any possible input objects supported by Python.
    :return: The same input objects.
    """
    time.sleep(5)
    return any_input


# Let's try to run the function on the same input twice. What would happen?
start_time = time.time()
print(expensive_function(2))  # 2
print(expensive_function(2))  # 2
print(f"Time for computation: {round(time.time()-start_time, 1)} seconds")  # Time for computation: 10.0 seconds.

# Does it make sense? We just did the same (slow) calculation (leading to the same output) twice!


# Let's wrap the function in a callable class, through the dunder __call__. Then, let's define a cache attribute!
class CachedExpensiveFunction:
    """
    A general-purpose caching class for storing and reusing the results of expensive computations.

    This class wraps a function and caches its results based on the inputs provided. When the function
    is called with a previously seen input, the cached result is returned instead of recomputing it.

    Attributes:
        func: The function whose results are to be cached.
        cache: A dictionary that maps inputs to their corresponding outputs.
    """

    def __init__(self, func: Callable) -> None:
        """Initializes the caching class with the provided func and an empty cache.

        :param func: The provided Callable
        :return: None
        """
        self.func = func
        self.cache = dict()

    def __call__(self, *args: tuple, **kwargs: dict):
        """
        Calls the wrapped 'func' with the given arguments. Check if the result is cached: if not, computes it, caches it
        and returns it, otherwise it just returns it from cache.

        :param args: Positional arguments for the callable attribute 'func'
        :param kwargs: Keyword arguments for the callable attribute 'func'
        :return: The result of the wrapped callable for the given arguments.
        """

        key = (tuple(args), tuple(sorted(kwargs.items())))

        if key not in self.cache:
            self.cache[key] = self.func(*args, **kwargs)

        return self.cache[key]


# Let's wrap 'expensive_function' on our new custom class. Then, let's call (__call__) it twice tracking the time.
cached_exp_func = CachedExpensiveFunction(func=expensive_function)
start_time = time.time()
print(cached_exp_func(2))  # 2
print(cached_exp_func(2))  # 2
print(f"Time for computation: {round(time.time()-start_time, 1)} seconds")  # Time for computation: 5.0 seconds.

# See? We got the same results, but we cut by half the execution time due to our caching system.

# What if we are interested in a local caching system? Well, we can reuse the same class just creating a new instance!
another_cached_exp_func = CachedExpensiveFunction(func=expensive_function)

start_time = time.time()
print(cached_exp_func(3))  # 3
print(another_cached_exp_func(3))  # 3
print(f"Time for computation: {round(time.time()-start_time, 1)} seconds")  # Time for computation: 10.0 seconds
