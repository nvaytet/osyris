# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2020 Osyris contributors (https://github.com/nvaytet/osyris)
# @author Neil Vaytet

import struct

def get_binary_data(fmt="",ninteg=0,nlines=0,nfloat=0,nstrin=0,nquadr=0,nlongi=0,offset=None,content=None,correction=0):
    """
    Determine binary offset when reading fortran binary files and return unpacked data
    """
    if offset is None:
        offset = 4*ninteg + 8*(nlines+nfloat+nlongi) + nstrin + nquadr*16
    offset += 4 + correction
    byte_size = {"b": 1 , "h": 2, "i": 4, "q": 8, "f": 4, "d": 8, "e": 8}
    if len(fmt) == 1:
        mult = 1
    else:
        mult = eval(fmt[0:len(fmt)-1])
    pack_size = mult*byte_size[fmt[-1]]

    return struct.unpack(fmt, content[offset:offset+pack_size])


def value_to_string(val, precision=3):
    """
    Convert number to string in well formatted manner.
    """
    if (not isinstance(val, float)) or (val == 0):
        text = str(val)
    elif (abs(val) >= 10.0**(precision+1)) or \
         (abs(val) <= 10.0**(-precision-1)):
        text = "{val:.{prec}e}".format(val=val, prec=precision)
    else:
        text = "{}".format(val)
        if len(text) > precision + 2 + (text[0] == '-'):
            text = "{val:.{prec}f}".format(val=val, prec=precision)
    return text
