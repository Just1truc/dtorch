# Copyright 2015 Jared Rodriguez (jared at blacknode dot net)
# All Rights Reserved.
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

"""
Taken from the press project
"""
import sys

from decimal import Decimal, InvalidOperation


__PYTHON_2 = sys.version_info[0] == 2


def check_int_type(o):
    if __PYTHON_2:
        # noinspection PyUnresolvedReferences
        return isinstance(o, (int, long))
    return isinstance(o, int)


def check_str_type(o):
    if __PYTHON_2:
        # noinspection PyUnresolvedReferences
        return isinstance(o, (str, unicode))
    return isinstance(o, str)


class SizeObjectValError(ValueError):
    pass


PY3 = sys.version_info[0] == 3

if PY3:
    string_types = str,
else:
    # noinspection PyCompatibility,PyUnresolvedReferences
    string_types = basestring,


class Size(object):
    byte = 1
    kibibyte = 1024
    # TODO: Make these lambda's so the references are not holding RAM
    mebibyte = kibibyte ** 2
    gibibyte = kibibyte ** 3
    tebibyte = kibibyte ** 4
    pebibyte = kibibyte ** 5
    exbibyte = kibibyte ** 6
    zebibyte = kibibyte ** 7
    yobibyte = kibibyte ** 8

    # Because we are dealing with disks, we'll probably need decimal byte notation

    kilobyte = 1000
    megabyte = kilobyte ** 2
    gigabyte = kilobyte ** 3
    terabyte = kilobyte ** 4
    petabyte = kilobyte ** 5
    exabyte = kilobyte ** 6
    zettabyte = kilobyte ** 7
    yottabyte = kilobyte ** 8

    sector = 512

    symbols = {
        'b': byte,
        'k': kilobyte,
        'kB': kilobyte,
        'KiB': kibibyte,
        'M': megabyte,
        'MB': megabyte,
        'MiB': mebibyte,
        'G': gigabyte,
        'GB': gigabyte,
        'GiB': gibibyte,
        'T': terabyte,
        'TB': terabyte,
        'TiB': tebibyte,
        'PB': petabyte,
        'PiB': pebibyte,
        'EB': exabyte,
        'EiB': exbibyte,
        'YB': yottabyte,
        'YiB': yobibyte,
        's': sector
    }

    iec_symbol_converter = {
        'k': 'KiB',
        'kB': 'KiB',
        'M': 'MiB',
        'MB': 'MiB',
        'G': 'GiB',
        'GB': 'GiB',
        'T': 'TiB',
        'TB': 'TiB',
        'PB': 'PiB'  # We'll top here... because yolo
    }

    def __init__(self, value, force_iec_values=False):
        """

        :param value:
        :param force_iec_values: Force IEC values even when the user enters Metric/JDEC notations
        :return:
        """
        self.force_iec_values = force_iec_values
        self.bytes = self._convert(value)

    def _convert(self, value):
        if isinstance(value, Size):
            return value.bytes

        if check_int_type(value):
            if value > self.yobibyte:
                raise SizeObjectValError('Value is impossibly large.')
            return value

        if isinstance(value, (float, Decimal)):
            return int(round(value))

        if not isinstance(value, string_types):
            raise SizeObjectValError(
                'Value is not in a format I can understand : {} - {}'.format(type(value), value))

        if value.isdigit():
            return int(value)

        valid_suffices = self.symbols.keys()
        suffix_index = 0
        for valid_suffix in valid_suffices:
            our_index = value.find(valid_suffix)
            if our_index and our_index != -1:
                suffix_index = our_index
                break
        if not suffix_index:
            raise SizeObjectValError(
                'Value is not in a format I can understand. Invalid Suffix. {}'.format(value))

        val, suffix = value[:suffix_index].strip(), value[suffix_index:].strip()

        if suffix not in valid_suffices:
            raise SizeObjectValError(
                'Value is not in a format I can understand. Invalid Suffix.')

        try:
            val = Decimal(val)
        except InvalidOperation:
            raise SizeObjectValError(
                'Value is not in a format I can understand. '
                'Could not convert value to int')

        if self.force_iec_values:
            suffix = self.iec_symbol_converter.get(suffix, suffix)
        return int(round(val * self.symbols[suffix]))

    @property
    def humanize(self):
        units = ['KiB', 'MiB', 'GiB', 'TiB', 'PiB', 'EiB', 'TiB']
        if self.bytes < self.symbols['KiB']:
            return '%d b' % self.bytes

        for idx in range(1, len(units)):
            if self.bytes < self.symbols[units[idx]]:
                unit = units[idx - 1]
                return '%s %s' % (round(Decimal(self.bytes) / self.symbols[unit]), unit)

        raise SizeObjectValError('Something very strange has happened.')

    @property
    def over_2t(self):
        if self.bytes >= self.tebibyte * 2:
            return True
        return False

    @property
    def megabytes(self):
        return Decimal(self.bytes) / self.megabyte

    @property
    def mebibytes(self):
        return Decimal(self.bytes) / self.mebibyte

    def __repr__(self):
        rep = '<%s> : %ib' % (self.__class__.__name__, self.bytes)
        return rep

    def __str__(self):
        return str(self.humanize)

    def __unicode__(self):
        return self.humanize

    def __add__(self, other):
        return Size(self.bytes + other.bytes)

    def __sub__(self, other):
        return Size(self.bytes - other.bytes)

    def __mul__(self, other):
        return Size(self.bytes * other.bytes)

    def __div__(self, other):
        'dont devide by zero.'
        return Size(self.bytes / other.bytes)

    def __mod__(self, other):
        return Size(self.bytes % other.bytes)

    def __lt__(self, other):
        return self.bytes < other.bytes

    def __le__(self, other):
        return self.bytes <= other.bytes

    def __eq__(self, other):
        return self.bytes == other.bytes

    def __ne__(self, other):
        return self.bytes != other.bytes

    def __gt__(self, other):
        return self.bytes > other.bytes

    def __ge__(self, other):
        return self.bytes >= other.bytes

    def __truedev__(self, other):
        'dont device by zero'
        return Size(other.bytes / self.bytes)


class PercentString(object):
    def __init__(self, our_string):
        """
        Examples:
            25% : 25% of entire container
            40%FREE : 40 percent of free space

        Precision is to the nearest 100th
        """

        self.raw_string = our_string
        self.free = False
        idx = self.raw_string.find('%')
        if idx == -1 or idx > 3:
            raise ValueError('Invalid expression')

        value, mod = self.raw_string.split('%')

        if not value.isdigit():
            raise ValueError('Invalid expression')

        value = int(value)
        if value > 100:
            raise ValueError('> 100%')

        self.value = value * .01

        if mod.upper() == 'FREE':
            self.free = True

    def __repr__(self):
        return self.raw_string
