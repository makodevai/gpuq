import sys
import types


class staticproperty(property):
    def __init__(self, fget=None, fset=None, fdel=None, doc=None):
        if fget is not None and not isinstance(fget, staticmethod):
            fget = staticmethod(fget)
        if fset is not None and not isinstance(fset, staticmethod):
            fset = staticmethod(fset)
        if fdel is not None and not isinstance(fdel, staticmethod):
            fdel = staticmethod(fdel)
        super().__init__(fget, fset, fdel, doc)

    def __get__(self, inst, cls=None):
        if inst is None:
            return self
        if self.fget is None:
            raise AttributeError("unreadable attribute")
        return self.fget.__get__(inst, cls)()  # pylint: disable=no-member

    def __set__(self, inst, val):
        if self.fset is None:
            raise AttributeError("can't set attribute")
        return self.fset.__get__(inst)(val)  # pylint: disable=no-member

    def __delete__(self, inst):
        if self.fdel is None:
            raise AttributeError("can't delete attribute")
        return self.fdel.__get__(inst)()  # pylint: disable=no-member


class LazyModuleType(types.ModuleType):
    def __init__(self, name):
        super().__init__(name)

    def __getattribute__(self, name: str):
        _props = super().__getattribute__("_props")
        if name in _props:
            return object.__getattribute__(self, name)
        else:
            return types.ModuleType.__getattribute__(self, name)

    def __dir__(self):
        ret = super().__dir__()
        ret.extend(self._props)
        return ret


def add_module_properties(module_name, properties):
    module = sys.modules[module_name]
    replace = False
    if isinstance(module, LazyModuleType):
        hacked_type = type(module)
    else:
        hacked_type = type(
            "LazyModuleType__{}".format(module_name.replace(".", "_")),
            (LazyModuleType,),
            {"_props": set()},
        )
        replace = True

    for name, prop in properties.items():
        if not isinstance(prop, property):
            prop = property(prop)
        setattr(hacked_type, name, prop)
        hacked_type._props.add(name)

    if replace:
        new_module = hacked_type(module_name)
        new_module.__dict__.update(module.__dict__)
        sys.modules[module_name] = new_module
