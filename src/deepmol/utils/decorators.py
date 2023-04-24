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
            kwargs["inplace"] = True
            method(result, *args, **kwargs)
            self.__dict__.update(result.__dict__)
            return self
    return inplace_method
