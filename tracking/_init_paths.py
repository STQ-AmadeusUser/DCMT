from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path as osp
import sys


def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)


this_dir = osp.dirname(__file__)

lib_path = osp.join(this_dir, '..', 'lib')
add_path(lib_path)

lib_path1 = osp.join(this_dir, '..', 'pytorch-grad-cam')
add_path(lib_path1)

lib_path2 = osp.join(this_dir, '..')
add_path(lib_path2)


