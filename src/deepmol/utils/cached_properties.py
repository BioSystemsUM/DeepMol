import cached_property


class deepmol_cached_property(cached_property.cached_property):
    def __set__(self, instance, value):
        raise AttributeError("can't set attribute")