# Copyright (c) 2024 SparseEnd2End. All rights reserved @author: Thomas Von Wu.
import re
import ast
import sys
import types
import tempfile
import os.path as osp
from importlib import import_module
from yapf.yapflib.yapf_api import FormatCode


def substitute_predefined_vars(filename, temp_config_name):
    file_dirname = osp.dirname(filename)
    file_basename = osp.basename(filename)
    file_basename_no_extension = osp.splitext(file_basename)[0]
    file_extname = osp.splitext(filename)[1]
    support_templates = dict(
        fileDirname=file_dirname,
        fileBasename=file_basename,
        fileBasenameNoExtension=file_basename_no_extension,
        fileExtname=file_extname,
    )
    with open(filename, encoding="utf-8") as f:
        # Setting encoding explicitly to resolve coding issue on windows
        config_file = f.read()
    for key, value in support_templates.items():
        regexp = r"\{\{\s*" + str(key) + r"\s*\}\}"
        value = value.replace("\\", "/")
        config_file = re.sub(regexp, value, config_file)
    with open(temp_config_name, "w", encoding="utf-8") as tmp_config_file:
        tmp_config_file.write(config_file)


def validate_py_syntax(filename):
    with open(filename, encoding="utf-8") as f:
        # Setting encoding explicitly to resolve coding issue on windows
        content = f.read()
    try:
        ast.parse(content)
    except SyntaxError as e:
        raise SyntaxError("There are syntax errors in config " f"file {filename}: {e}")


def file2dict(filename, use_predefined_variables=True):
    filename = osp.abspath(osp.expanduser(filename))
    fileExtname = osp.splitext(filename)[1]
    assert fileExtname == ".py"

    with tempfile.TemporaryDirectory() as temp_config_dir:
        temp_config_file = tempfile.NamedTemporaryFile(
            dir=temp_config_dir, suffix=fileExtname
        )
        temp_config_name = osp.basename(temp_config_file.name)
        # Substitute predefined variables
        if use_predefined_variables:
            substitute_predefined_vars(filename, temp_config_file.name)

        temp_module_name = osp.splitext(temp_config_name)[0]
        sys.path.insert(0, temp_config_dir)
        validate_py_syntax(filename)
        mod = import_module(temp_module_name)
        sys.path.pop(0)
        cfg_dict = {
            name: value
            for name, value in mod.__dict__.items()
            if not name.startswith("__")
            and not isinstance(value, types.ModuleType)
            and not isinstance(value, types.FunctionType)
        }
        # delete imported module
        del sys.modules[temp_module_name]

        # close temp file
        temp_config_file.close()

    return cfg_dict


def read_cfg(filename, use_predefined_variables=True, import_custom_modules=True):
    cfg_dict = file2dict(filename, use_predefined_variables)
    return cfg_dict


import addict

from typing import Dict


class ConfigDict(addict.Dict):
    def __missing__(self, name):
        raise KeyError(name)

    def __getattr__(self, name):
        try:
            value = super().__getattr__(name)
        except KeyError:
            ex = AttributeError(
                f"'{self.__class__.__name__}' object has no " f"attribute '{name}'"
            )
        except Exception as e:
            ex = e
        else:
            return value
        raise ex


def pretty_text(cfg_dict: Dict):
    """
    Convert traditional dict datatype to addict.Dict object.
    format Dict object and get pretty text(str).
    """
    cfg_dict = ConfigDict(cfg_dict)
    indent = 4

    def _indent(s_, num_spaces):
        s = s_.split("\n")
        if len(s) == 1:
            return s_
        first = s.pop(0)
        s = [(num_spaces * " ") + line for line in s]
        s = "\n".join(s)
        s = first + "\n" + s
        return s

    def _format_basic_types(k, v, use_mapping=False):
        if isinstance(v, str):
            v_str = f"'{v}'"
        else:
            v_str = str(v)

        if use_mapping:
            k_str = f"'{k}'" if isinstance(k, str) else str(k)
            attr_str = f"{k_str}: {v_str}"
        else:
            attr_str = f"{str(k)}={v_str}"
        attr_str = _indent(attr_str, indent)

        return attr_str

    def _format_list(k, v, use_mapping=False):
        # check if all items in the list are dict
        if all(isinstance(_, dict) for _ in v):
            v_str = "[\n"
            v_str += "\n".join(
                f"dict({_indent(_format_dict(v_), indent)})," for v_ in v
            ).rstrip(",")
            if use_mapping:
                k_str = f"'{k}'" if isinstance(k, str) else str(k)
                attr_str = f"{k_str}: {v_str}"
            else:
                attr_str = f"{str(k)}={v_str}"
            attr_str = _indent(attr_str, indent) + "]"
        else:
            attr_str = _format_basic_types(k, v, use_mapping)
        return attr_str

    def _contain_invalid_identifier(dict_str):
        contain_invalid_identifier = False
        for key_name in dict_str:
            contain_invalid_identifier |= not str(key_name).isidentifier()
        return contain_invalid_identifier

    def _format_dict(input_dict, outest_level=False):
        r = ""
        s = []

        use_mapping = _contain_invalid_identifier(input_dict)
        if use_mapping:
            r += "{"
        for idx, (k, v) in enumerate(input_dict.items()):
            is_last = idx >= len(input_dict) - 1
            end = "" if outest_level or is_last else ","
            if isinstance(v, dict):
                v_str = "\n" + _format_dict(v)
                if use_mapping:
                    k_str = f"'{k}'" if isinstance(k, str) else str(k)
                    attr_str = f"{k_str}: dict({v_str}"
                else:
                    attr_str = f"{str(k)}=dict({v_str}"
                attr_str = _indent(attr_str, indent) + ")" + end
            elif isinstance(v, list):
                attr_str = _format_list(k, v, use_mapping) + end
            else:
                attr_str = _format_basic_types(k, v, use_mapping) + end

            s.append(attr_str)
        r += "\n".join(s)
        if use_mapping:
            r += "}"
        return r

    cfg_dict = cfg_dict.to_dict()
    text = _format_dict(cfg_dict, outest_level=True)
    # copied from setup.cfg
    yapf_style = dict(
        based_on_style="pep8",
        blank_line_before_nested_class_or_def=True,
        split_before_expression_after_opening_paren=True,
    )
    text, _ = FormatCode(text, style_config=yapf_style, verify=True)
    return text
