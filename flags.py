"""Implementation of the flags interface."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse as _argparse

_global_parser = _argparse.ArgumentParser()


class _FlagValues(object):
	"""Global container and accessor for flags and their values."""

	def __init__(self):
		self.__dict__['__flags'] = {}
		self.__dict__['__parsed'] = False

	def _parse_flags(self, args=None):
		result, unparsed = _global_parser.parse_known_args(args=args)
		for flag_name, val in vars(result).items():
			self.__dict__['__flags'][flag_name] = val
		self.__dict__['__parsed'] = True
		return unparsed

	def __getattr__(self, name):
		"""Retrieves the 'value' attribute of the flag --name."""
		if not self.__dict__['__parsed']:
			self._parse_flags()
		if name not in self.__dict__['__flags']:
			raise AttributeError(name)
		return self.__dict__['__flags'][name]

	def __setattr__(self, name, value):
		"""Sets the 'value' attribute of the flag --name."""
		if not self.__dict__['__parsed']:
			self._parse_flags()
		self.__dict__['__flags'][name] = value


def _define_helper(flag_name, default_value, docstring, flagtype):
	"""Registers 'flag_name' with 'default_value' and 'docstring'."""
	_global_parser.add_argument('--' + flag_name,
															default=default_value,
															help=docstring,
															type=flagtype)


# Provides the global object that can be used to access flags.
FLAGS = _FlagValues()


def DEFINE_string(flag_name, default_value, docstring):
	_define_helper(flag_name, default_value, docstring, str)


def DEFINE_integer(flag_name, default_value, docstring):
	_define_helper(flag_name, default_value, docstring, int)


def DEFINE_boolean(flag_name, default_value, docstring):
	def str2bool(v):
		return v.lower() in ('true', 't', '1')
	_global_parser.add_argument('--' + flag_name,
															nargs='?',
															const=True,
															help=docstring,
															default=default_value,
															type=str2bool)

	# Add negated version, stay consistent with argparse with regard to
	# dashes in flag names.
	_global_parser.add_argument('--no' + flag_name,
															action='store_false',
															dest=flag_name.replace('-', '_'))


# The internal google library defines the following alias, so we match
# the API for consistency.
DEFINE_bool = DEFINE_boolean	# pylint: disable=invalid-name


def DEFINE_float(flag_name, default_value, docstring):
	_define_helper(flag_name, default_value, docstring, float)

def DEFINE_list(flag_name, default_value, docstring):
	_define_helper(flag_name, default_value, docstring, list)

