def inplace_decorator(method: callable) -> callable:
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
        if inplace:
            method(self, *args, **kwargs)
            return None
        else:
            result = self.__copy__()
            method(result, *args, **kwargs)
            return result
    return inplace_method
