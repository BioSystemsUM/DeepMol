from concurrent import futures
from copy import deepcopy
from typing import Union


def inplace_decorator(method: callable) -> Union[callable, None]:
    """
    Decorator to make a method inplace.

    Parameters
    ----------
    method: callable
        Method to decorate.

    Returns
    -------
    inplace_method: callable
        Decorated method.
    """
    def inplace_method(self, *args, inplace=False, **kwargs):
        """
        Method to make inplace.

        Parameters
        ----------
        self: object
            Object to apply the method to.
        args: list
            Arguments to pass to the method.
        inplace: bool
            Whether to apply the method inplace.
        kwargs: dict
            Keyword arguments to pass to the method.

        Returns
        -------
        result: object
            Result of the method.
        """
        if inplace:
            method(self, *args, **kwargs)
            return None
        else:
            result = deepcopy(self)
            method(result, *args, **kwargs)
            return result
    return inplace_method


def modify_object_inplace_decorator(method: callable) -> Union[callable, None]:
    """
    Decorator to create a lazy copy-on-write version of a method.

    This decorator performs modifications of an object that is received by the class, either inplace or on a copy of the
    object, depending on the value of the `inplace` parameter.


    This applies inplace

    Parameters
    ----------
    method: callable
        The method to decorate.

    Returns
    -------
    modify_object_wrapper: callable
        The decorated method.
    """
    def modify_object_wrapper(self, other_object, inplace=False, **kwargs):
        """
        Method that modifies an input object inplace or on a copy.

        Parameters
        ----------
        self: object
            The class instance object.
        other_object: object
            The object to apply the method to.
        inplace: bool
            Whether to apply the method in place.
        kwargs: dict
            Keyword arguments to pass to the method.

        Returns
        -------
        new_object: object
            The new object.
        """
        if inplace:
            # modify the other_object in-place
            method(self, other_object, **kwargs)
            return None
        else:
            # create a new copy of the other_object
            new_object = deepcopy(other_object)
            method(self, new_object, **kwargs)
            return new_object
    return modify_object_wrapper


def timeout(timelimit):
    def decorator(func):
        def decorated(*args, **kwargs):
            with futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(func, *args, **kwargs)
                try:
                    result = future.result(timelimit)
                except futures.TimeoutError:
                    raise TimeoutError(f"Function {func.__name__} timed out after {timelimit} seconds.")
                executor._threads.clear()
                futures.thread._threads_queues.clear()
                return result
        return decorated
    return decorator
