import pandas as pd
from copy import deepcopy
import numpy as np


class Struct():
    '''A class for nested dictionary with basic operations'''
    def __init__(self):
        self.ndict = {}
        self.n_layers = 0

    def add_dict_layer(self, keys):
        assert isinstance(keys, list)
        self.add_layer(keys, values=dict)

    def add_layer(self, keys, values=None):
        assert values is None or type(values) is type

        def add_rec(struct, level):
            if level == self.n_layers:
                if values is None:
                    vals = [None] * len(keys)
                else:
                    vals = [values() for i in range(len(keys))]
                struct.update(dict(zip(keys, vals)))
            else:
                for key in struct:
                    add_rec(struct[key], level+1)
        add_rec(self.ndict, 0)
        self.n_layers += 1

    @classmethod
    def new_empty(cls, key_map, values=None):
        struct = cls()
        for keys in key_map[:-1]:
            struct.add_dict_layer(keys)
        struct.add_layer(key_map[-1], values)
        return struct

    @classmethod
    def from_nested_dict(cls, nested_dict, n_layers):
        struct = cls()
        struct.ndict = nested_dict
        struct.n_layers = n_layers
        return struct

    @classmethod
    def empty_like(cls, struct, values=None):
        key_map = struct.get_key_map()
        return cls.new_empty(key_map, values=values)

    def set_all_leaf(self, value):
        key_map = self.get_key_map()

        def set_rec(key_seq, lvl):
            if lvl == self.n_layers:
                self.__setitem__(key_seq, value)
            else:
                lvl_keys = key_map[lvl]
                key_seq = key_seq.copy()
                key_seq.append(None)
                for key in lvl_keys:
                    key_seq[-1] = key
                    set_rec(key_seq, lvl+1)
        set_rec([], 0)

    def apply_agg(self, func, depth=None, **kwargs):
        res = self.apply(func, depth=depth, **kwargs)
        res.n_layers = depth
        return res

    def apply_agg_(self, *args, **kwargs):
        res = self.apply_agg(*args, **kwargs)
        self.ndict = res.ndict
        self.n_layers = res.n_layers

    def combine(self, other):
        assert isinstance(other, Struct)

        kmap1 = self.get_key_map()
        kmap2 = other.get_key_map()
        newkmap = []
        mismatch = False
        for depth in range(self.n_layers):
            if kmap1[depth] != kmap2[depth]:
                if mismatch:
                    raise ValueError('Structures must have different keymaps on a single level only')
                else:
                    mismatch = True
            newkmap.append(list(set(kmap1[depth]+kmap2[depth])))

        def comb_rec(ndict1, ndict2, level):
            if level == self.n_layers:
                if ndict1 is not None:
                    return ndict1
                else:
                    return ndict2
            else:
                res = {}
                for key in newkmap[level]:
                    if ndict1 and key in ndict1:
                        ndict1next = ndict1[key]
                    else:
                        ndict1next = None
                    if ndict2 and key in ndict2:
                        ndict2next = ndict2[key]
                    else:
                        ndict2next = None
                    res[key] = comb_rec(ndict1next, ndict2next, level+1)
                return res

        struct = Struct.from_nested_dict(comb_rec(self.ndict, other.ndict, 0), self.n_layers)
        return struct

    def apply(self, func, depth=None, args=(), kwargs={}):
        assert depth is None or isinstance(depth, int)
        if not depth:
            depth = self.n_layers

        def apply_rec(ndict, level):
            if level == depth or not isinstance(ndict, dict):
                return func(ndict, *args, **kwargs)
            else:
                res = {}
                for key in ndict:
                    res[key] = apply_rec(ndict[key], level+1)
                return res

        struct = Struct.from_nested_dict(apply_rec(self.ndict, 0), depth)
        return struct

    def apply_(self, *args, **kwargs):
        struct = self.apply(*args, **kwargs)
        self.ndict = struct.ndict
        self.n_layers = struct.n_layers

    def __repr__(self):
        key_map = self.get_key_map()
        type = self.get_type()
        first_item = self.get_first_item()
        string = "first item:\n" + str(first_item)
        if isinstance(first_item, np.ndarray):
            string += '\narray shape: ' + str(first_item.shape)
        if isinstance(first_item, list):
            string += '\nlist length: ' + str(len(first_item))
        string += '\nstructure:\n' + str(key_map) + str(type)

        return string

    def __getitem__(self, key_seq):
        if key_seq.__class__ not in [list, tuple]:
            assert key_seq in self.ndict.keys()
            key_seq = (key_seq,)
        else:
            assert len(key_seq) <= self.n_layers
        lvl = 0
        ndict = self.ndict
        while lvl < len(key_seq):
            ndict = ndict[key_seq[lvl]]
            lvl += 1
        return ndict

    def get_first_item(self):
        key_map = self.get_key_map()
        lvl = 0
        key_seq = []
        while lvl < self.n_layers:
            key_seq.append(key_map[lvl][0])
            lvl += 1
        return self.__getitem__(key_seq)

    def __setitem__(self, key_seq, value):
        if key_seq.__class__ not in [list, tuple]:
            assert key_seq in self.ndict.keys()
            key_seq = (key_seq,)
        else:
            assert len(key_seq) <= self.n_layers
        lvl = 0
        ndict = self.ndict
        while lvl < self.n_layers - 1:
            ndict = ndict[key_seq[lvl]]
            lvl += 1
        ndict[key_seq[-1]] = value

    def get_key_map(self):
        lvl = 0
        key_map = []
        ndict = self.ndict
        while lvl < self.n_layers:
            keys = list(ndict.keys())
            key_map.append(keys)
            ndict = ndict[keys[0]]
            lvl += 1
        return key_map

    def get_type(self):
        key_map = self.get_key_map()
        key_seq = [lvl_keys[0] for lvl_keys in key_map]
        return type(self.__getitem__(key_seq))

    def _reorder_levels(self, new_key_map, reorder_func, reorder_args):
        key_map = self.get_key_map()
        struct = self.__class__.new_empty(new_key_map)

        def move_rec(key_seq, lvl):
            if lvl == self.n_layers:
                new_key_seq = reorder_func(key_seq, *reorder_args)
                struct.__setitem__(new_key_seq, self.__getitem__(key_seq))
            else:
                lvl_keys = key_map[lvl]
                key_seq = key_seq.copy()
                key_seq.append(None)
                for key in lvl_keys:
                    key_seq[-1] = key
                    move_rec(key_seq, lvl+1)

        move_rec([], 0)
        return struct

    def _swap_seq(self, seq, lvl1, lvl2):
        seq = list(seq)
        seq_new = seq.copy()
        key_lvl2, key_lvl1 = seq_new.pop(lvl2), seq_new.pop(lvl1)
        seq_new.insert(lvl1, key_lvl2)
        seq_new.insert(lvl2, key_lvl1)
        return seq_new

    def _move_seq(self, seq, lvl, position):
        seq = list(seq)
        seq_new = seq.copy()
        key_lvl = seq_new.pop(lvl)
        seq_new.insert(position, key_lvl)
        return seq_new

    def _swap_key_map(self, lvl1, lvl2):
        key_map = self.get_key_map()
        return self._swap_seq(key_map, lvl1, lvl2)

    def _move_key_map(self, lvl, position):
        key_map = self.get_key_map()
        return self._move_seq(key_map, lvl, position)

    def swap_levels(self, lvl1, lvl2):
        assert isinstance(lvl1, int) and isinstance(lvl2, int)
        assert lvl1 < self.n_layers and lvl2 < self.n_layers
        assert lvl1 != lvl2
        lvl1, lvl2 = sorted([lvl1, lvl2])
        key_map_new = self._swap_key_map(lvl1, lvl2)
        return self._reorder_levels(key_map_new, self._swap_seq, (lvl1, lvl2))

    def swap_levels_(self, *args):
        self.ndict = self.swap_levels(*args).ndict

    def move_level(self, lvl, position):
        assert isinstance(lvl, int) and isinstance(position, int)
        assert lvl < self.n_layers and position < self.n_layers
        assert lvl != position
        key_map_new = self._move_key_map(lvl, position)
        return self._reorder_levels(key_map_new, self._move_seq, (lvl, position))

    def move_level_(self, *args):
        self.ndict = self.move_level(*args).ndict

    def to_pandas(self, columns=None):
        def rec(ndict):
            if not isinstance(ndict, dict):
                return [[ndict]]
            else:
                newrows = []
                for key, val in ndict.items():
                    for row in rec(val):
                        newrows.append([key] + row)
                return newrows
        rows = rec(self.ndict)
        return pd.DataFrame.from_records(rows, columns=columns)

    def to_list(self):
        def list_rec(ndict, level, out_list):
            if level == self.n_layers:
                return out_list + [ndict]
            else:
                for key in ndict:
                    out_list = list_rec(ndict[key], level+1, out_list)
                return out_list

        out_list = []
        return list_rec(self.ndict, 0, out_list)

    def copy(self):
        ndict_new = deepcopy(self.ndict)
        return self.__class__.from_nested_dict(ndict_new, self.n_layers)

    def restrict_level(self, level, value, inplace=False):
        '''Restrict a level to a particular key, removing data from other keys'''
        assert level < self.n_layers
        if level < self.n_layers - 1:
            self.move_level_(level, self.n_layers - 1)
        key_map = self.get_key_map()
        if inplace:
            # If in place, a new instance sharing the reference of the original
            # instance is created. The new instance has one less level which
            # allows to use __setitem__ at the second to last level while
            # modifying the original instance
            struct = self.__class__.from_nested_dict(self.ndict, self.n_layers-1)
        else:
            struct = self.__class__.new_empty(key_map[:-1])

        def restrict_rec(key_seq, lvl):
            if lvl == self.n_layers - 1:
                struct.__setitem__(key_seq, self.__getitem__(key_seq + [value]))
            else:
                lvl_keys = key_map[lvl]
                key_seq = key_seq.copy()
                key_seq.append(None)
                for key in lvl_keys:
                    key_seq[-1] = key
                    restrict_rec(key_seq, lvl+1)

        restrict_rec([], 0)
        if inplace:
            self.n_layers -= 1
        else:
            return struct

    def restrict_level_(self, *args):
        self.restrict_level(*args, inplace=True)

    # def iterate_last_layer(self):
    #     ilayer = 0
    #     while ilayer < nlayers:

    def __add__(self, other):
        if not isinstance(other, Struct):
            raise TypeError("Only a Struct can be added to a Struct to zip the leafs of the structs...")
        key_map = self.get_key_map()
        struct = self.new_empty(self.get_key_map())

        def add_rec(key_seq, lvl):
            if lvl == self.n_layers:
                struct.__setitem__(key_seq, (self.__getitem__(key_seq), other.__getitem__(key_seq)))
            else:
                lvl_keys = key_map[lvl]
                key_seq = key_seq.copy()
                for key in lvl_keys:
                    add_rec(key_seq + [key], lvl+1)

        add_rec([], 0)
        return struct


if __name__ == '__main__':
    struct = Struct.new_empty([['a', 'b'], [1, 2, 3], ['bloup', 'blop']])
    struct.set_all_leaf(42)
    struct.to_list()
    struct2 = Struct.empty_like(struct)
    struct2.set_all_leaf(43)
    struct3 = struct + struct2
    struct3.to_list()

    struct3 = Struct.new_empty([['a', 'b'], [4], ['bloup', 'blop']])
    struct3.set_all_leaf(43)

    struct.combine(struct3)
