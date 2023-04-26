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
            result = self.__copy__()
            method(result, *args, **kwargs)
            return result
    return inplace_method


def copy_on_write_decorator(method: callable) -> Union[callable, None]:
    """
    Decorator to make a method copy-on-write.

    Parameters
    ----------
    method: callable
        Method to decorate.

    Returns
    -------
    new_func: callable
        Decorated method.
    """
    def new_func(self, other_object, inplace=False, **kwargs):
        """
        Method to make copy-on-write.

        Parameters
        ----------
        self: object
            the class instance object
        other_object: object
            Object to apply the method to.
        inplace: bool
            Whether to apply the method inplace.
        kwargs: dict
            Keyword arguments to pass to the method.

        Returns
        -------
        new_object: object
            New object.
        """
        if inplace:
            # modify the other_object in-place
            method(self, other_object, **kwargs)
            return None
        else:
            # create a new copy of the other_object
            new_object = other_object.__copy__()
            method(self, new_object, **kwargs)
            return new_object
    return new_func
