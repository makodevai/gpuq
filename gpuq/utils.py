import os
import sys
import types
from typing import Any, Callable, Iterable


class staticproperty(property):
    def __init__(
        self,
        fget: Callable[[], Any] | None = None,
        fset: Callable[[Any], None] | None = None,
        fdel: Callable[[], None] | None = None,
        doc: str | None = None,
    ) -> None:
        if fget is not None and not isinstance(fget, staticmethod):
            fget = staticmethod(fget)
        if fset is not None and not isinstance(fset, staticmethod):
            fset = staticmethod(fset)
        if fdel is not None and not isinstance(fdel, staticmethod):
            fdel = staticmethod(fdel)
        super().__init__(fget, fset, fdel, doc)  # type: ignore

    def __get__(self, inst: Any, cls: type | None = None) -> Any:
        if inst is None:
            return self
        if self.fget is None:
            raise AttributeError("unreadable attribute")
        return self.fget.__get__(inst, cls)()  # pylint: disable=no-member

    def __set__(self, inst: Any, val: Any) -> None:
        if self.fset is None:
            raise AttributeError("can't set attribute")

        # pylint: disable=no-member
        return self.fset.__get__(inst)(val)  # type: ignore

    def __delete__(self, inst: Any) -> None:
        if self.fdel is None:
            raise AttributeError("can't delete attribute")
        # pylint: disable=no-member
        return self.fdel.__get__(inst)()  # type: ignore


class LazyModuleType(types.ModuleType):
    def __init__(self, name: str) -> None:
        super().__init__(name)

    def __getattribute__(self, name: str) -> Any:
        _props = super().__getattribute__("_props")
        if name in _props:
            return object.__getattribute__(self, name)
        else:
            return types.ModuleType.__getattribute__(self, name)

    def __dir__(self) -> Iterable[str]:
        ret = super().__dir__()
        ret.extend(self._props)  # type: ignore
        return ret


def add_module_properties(module_name: str, properties: dict[str, Any]) -> None:
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
        hacked_type._props.add(name)  # type: ignore

    if replace:
        new_module = hacked_type(module_name)
        module.__class__ = new_module.__class__
        module.__name__ = new_module.__name__
        module.__dict__.update(new_module.__dict__)


def int_or_none(value: str) -> int | None:
    if value.lower() in ["", "none", "no"]:
        return None
    return int(value)


def int_list(value: str) -> list[int]:
    return [int(el) for el in value.split(",")]


class default:
    def __init__(
        self,
        value: Any,
        env: str | None = None,
        env_parser: Callable[[str], Any] | None = None,
    ):
        self.value = value
        self.env = env
        self.env_parser = env_parser

    def get(self) -> Any:
        ret = self.value
        if self.env is not None:
            if self.env in os.environ:
                ret = os.environ[self.env]
                if self.env_parser is not None:
                    ret = self.env_parser(ret)

        return ret
