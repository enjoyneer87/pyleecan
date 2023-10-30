# -*- coding: utf-8 -*-
# File generated according to Generator/ClassesRef/Mesh/Interpolation/Interpolation.csv
# WARNING! All changes made in this file will be lost!
"""Method code available at https://github.com/Eomys/pyleecan/tree/master/pyleecan/Methods/Mesh/Interpolation
"""

from os import linesep
from sys import getsizeof
from logging import getLogger
from ._check import check_var, raise_
from ..Functions.get_logger import get_logger
from ..Functions.save import save
from ..Functions.load import load_init_dict
from ..Functions.Load.import_class import import_class
from copy import deepcopy
from ._frozen import FrozenClass

from numpy import isnan
from ._check import InitUnKnowClassError


class Interpolation(FrozenClass):
    """Store shape functions"""

    VERSION = 1

    # generic save method is available in all object
    save = save
    # get_logger method is available in all object
    get_logger = get_logger

    def __init__(
        self,
        ref_element=None,
        gauss_point=None,
        scalar_product=None,
        init_dict=None,
        init_str=None,
    ):
        """Constructor of the class. Can be use in three ways :
        - __init__ (arg1 = 1, arg3 = 5) every parameters have name and default values
            for pyleecan type, -1 will call the default constructor
        - __init__ (init_dict = d) d must be a dictionary with property names as keys
        - __init__ (init_str = s) s must be a string
        s is the file path to load

        ndarray or list can be given for Vector and Matrix
        object or dict can be given for pyleecan Object"""

        if init_str is not None:  # Load from a file
            init_dict = load_init_dict(init_str)[1]
        if init_dict is not None:  # Initialisation by dict
            assert type(init_dict) is dict
            # Overwrite default value with init_dict content
            if "ref_element" in list(init_dict.keys()):
                ref_element = init_dict["ref_element"]
            if "gauss_point" in list(init_dict.keys()):
                gauss_point = init_dict["gauss_point"]
            if "scalar_product" in list(init_dict.keys()):
                scalar_product = init_dict["scalar_product"]
        # Set the properties (value check and convertion are done in setter)
        self.parent = None
        self.ref_element = ref_element
        self.gauss_point = gauss_point
        self.scalar_product = scalar_product

        # The class is frozen, for now it's impossible to add new properties
        self._freeze()

    def __str__(self):
        """Convert this object in a readeable string (for print)"""

        Interpolation_str = ""
        if self.parent is None:
            Interpolation_str += "parent = None " + linesep
        else:
            Interpolation_str += (
                "parent = " + str(type(self.parent)) + " object" + linesep
            )
        if self.ref_element is not None:
            tmp = (
                self.ref_element.__str__().replace(linesep, linesep + "\t").rstrip("\t")
            )
            Interpolation_str += "ref_element = " + tmp
        else:
            Interpolation_str += "ref_element = None" + linesep + linesep
        if self.gauss_point is not None:
            tmp = (
                self.gauss_point.__str__().replace(linesep, linesep + "\t").rstrip("\t")
            )
            Interpolation_str += "gauss_point = " + tmp
        else:
            Interpolation_str += "gauss_point = None" + linesep + linesep
        if self.scalar_product is not None:
            tmp = (
                self.scalar_product.__str__()
                .replace(linesep, linesep + "\t")
                .rstrip("\t")
            )
            Interpolation_str += "scalar_product = " + tmp
        else:
            Interpolation_str += "scalar_product = None" + linesep + linesep
        return Interpolation_str

    def __eq__(self, other):
        """Compare two objects (skip parent)"""

        if type(other) != type(self):
            return False
        if other.ref_element != self.ref_element:
            return False
        if other.gauss_point != self.gauss_point:
            return False
        if other.scalar_product != self.scalar_product:
            return False
        return True

    def compare(self, other, name="self", ignore_list=None, is_add_value=False):
        """Compare two objects and return list of differences"""

        if ignore_list is None:
            ignore_list = list()
        if type(other) != type(self):
            return ["type(" + name + ")"]
        diff_list = list()
        if (other.ref_element is None and self.ref_element is not None) or (
            other.ref_element is not None and self.ref_element is None
        ):
            diff_list.append(name + ".ref_element None mismatch")
        elif self.ref_element is not None:
            diff_list.extend(
                self.ref_element.compare(
                    other.ref_element,
                    name=name + ".ref_element",
                    ignore_list=ignore_list,
                    is_add_value=is_add_value,
                )
            )
        if (other.gauss_point is None and self.gauss_point is not None) or (
            other.gauss_point is not None and self.gauss_point is None
        ):
            diff_list.append(name + ".gauss_point None mismatch")
        elif self.gauss_point is not None:
            diff_list.extend(
                self.gauss_point.compare(
                    other.gauss_point,
                    name=name + ".gauss_point",
                    ignore_list=ignore_list,
                    is_add_value=is_add_value,
                )
            )
        if (other.scalar_product is None and self.scalar_product is not None) or (
            other.scalar_product is not None and self.scalar_product is None
        ):
            diff_list.append(name + ".scalar_product None mismatch")
        elif self.scalar_product is not None:
            diff_list.extend(
                self.scalar_product.compare(
                    other.scalar_product,
                    name=name + ".scalar_product",
                    ignore_list=ignore_list,
                    is_add_value=is_add_value,
                )
            )
        # Filter ignore differences
        diff_list = list(filter(lambda x: x not in ignore_list, diff_list))
        return diff_list

    def __sizeof__(self):
        """Return the size in memory of the object (including all subobject)"""

        S = 0  # Full size of the object
        S += getsizeof(self.ref_element)
        S += getsizeof(self.gauss_point)
        S += getsizeof(self.scalar_product)
        return S

    def as_dict(self, type_handle_ndarray=0, keep_function=False, **kwargs):
        """
        Convert this object in a json serializable dict (can be use in __init__).
        type_handle_ndarray: int
            How to handle ndarray (0: tolist, 1: copy, 2: nothing)
        keep_function : bool
            True to keep the function object, else return str
        Optional keyword input parameter is for internal use only
        and may prevent json serializability.
        """

        Interpolation_dict = dict()
        if self.ref_element is None:
            Interpolation_dict["ref_element"] = None
        else:
            Interpolation_dict["ref_element"] = self.ref_element.as_dict(
                type_handle_ndarray=type_handle_ndarray,
                keep_function=keep_function,
                **kwargs
            )
        if self.gauss_point is None:
            Interpolation_dict["gauss_point"] = None
        else:
            Interpolation_dict["gauss_point"] = self.gauss_point.as_dict(
                type_handle_ndarray=type_handle_ndarray,
                keep_function=keep_function,
                **kwargs
            )
        if self.scalar_product is None:
            Interpolation_dict["scalar_product"] = None
        else:
            Interpolation_dict["scalar_product"] = self.scalar_product.as_dict(
                type_handle_ndarray=type_handle_ndarray,
                keep_function=keep_function,
                **kwargs
            )
        # The class name is added to the dict for deserialisation purpose
        Interpolation_dict["__class__"] = "Interpolation"
        return Interpolation_dict

    def copy(self):
        """Creates a deepcopy of the object"""

        # Handle deepcopy of all the properties
        if self.ref_element is None:
            ref_element_val = None
        else:
            ref_element_val = self.ref_element.copy()
        if self.gauss_point is None:
            gauss_point_val = None
        else:
            gauss_point_val = self.gauss_point.copy()
        if self.scalar_product is None:
            scalar_product_val = None
        else:
            scalar_product_val = self.scalar_product.copy()
        # Creates new object of the same type with the copied properties
        obj_copy = type(self)(
            ref_element=ref_element_val,
            gauss_point=gauss_point_val,
            scalar_product=scalar_product_val,
        )
        return obj_copy

    def _set_None(self):
        """Set all the properties to None (except pyleecan object)"""

        if self.ref_element is not None:
            self.ref_element._set_None()
        if self.gauss_point is not None:
            self.gauss_point._set_None()
        if self.scalar_product is not None:
            self.scalar_product._set_None()

    def _get_ref_element(self):
        """getter of ref_element"""
        return self._ref_element

    def _set_ref_element(self, value):
        """setter of ref_element"""
        if isinstance(value, str):  # Load from file
            try:
                value = load_init_dict(value)[1]
            except Exception as e:
                self.get_logger().error(
                    "Error while loading " + value + ", setting None instead"
                )
                value = None
        if isinstance(value, dict) and "__class__" in value:
            class_obj = import_class(
                "pyleecan.Classes", value.get("__class__"), "ref_element"
            )
            value = class_obj(init_dict=value)
        elif type(value) is int and value == -1:  # Default constructor
            RefElement = import_class("pyleecan.Classes", "RefElement", "ref_element")
            value = RefElement()
        check_var("ref_element", value, "RefElement")
        self._ref_element = value

        if self._ref_element is not None:
            self._ref_element.parent = self

    ref_element = property(
        fget=_get_ref_element,
        fset=_set_ref_element,
        doc=u"""

        :Type: RefElement
        """,
    )

    def _get_gauss_point(self):
        """getter of gauss_point"""
        return self._gauss_point

    def _set_gauss_point(self, value):
        """setter of gauss_point"""
        if isinstance(value, str):  # Load from file
            try:
                value = load_init_dict(value)[1]
            except Exception as e:
                self.get_logger().error(
                    "Error while loading " + value + ", setting None instead"
                )
                value = None
        if isinstance(value, dict) and "__class__" in value:
            class_obj = import_class(
                "pyleecan.Classes", value.get("__class__"), "gauss_point"
            )
            value = class_obj(init_dict=value)
        elif type(value) is int and value == -1:  # Default constructor
            GaussPoint = import_class("pyleecan.Classes", "GaussPoint", "gauss_point")
            value = GaussPoint()
        check_var("gauss_point", value, "GaussPoint")
        self._gauss_point = value

        if self._gauss_point is not None:
            self._gauss_point.parent = self

    gauss_point = property(
        fget=_get_gauss_point,
        fset=_set_gauss_point,
        doc=u"""

        :Type: GaussPoint
        """,
    )

    def _get_scalar_product(self):
        """getter of scalar_product"""
        return self._scalar_product

    def _set_scalar_product(self, value):
        """setter of scalar_product"""
        if isinstance(value, str):  # Load from file
            try:
                value = load_init_dict(value)[1]
            except Exception as e:
                self.get_logger().error(
                    "Error while loading " + value + ", setting None instead"
                )
                value = None
        if isinstance(value, dict) and "__class__" in value:
            class_obj = import_class(
                "pyleecan.Classes", value.get("__class__"), "scalar_product"
            )
            value = class_obj(init_dict=value)
        elif type(value) is int and value == -1:  # Default constructor
            ScalarProduct = import_class(
                "pyleecan.Classes", "ScalarProduct", "scalar_product"
            )
            value = ScalarProduct()
        check_var("scalar_product", value, "ScalarProduct")
        self._scalar_product = value

        if self._scalar_product is not None:
            self._scalar_product.parent = self

    scalar_product = property(
        fget=_get_scalar_product,
        fset=_set_scalar_product,
        doc=u"""

        :Type: ScalarProduct
        """,
    )
