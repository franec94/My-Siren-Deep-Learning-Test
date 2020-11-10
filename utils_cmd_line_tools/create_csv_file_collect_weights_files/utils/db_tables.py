from pprint import pprint

import copy
import contextlib
import collections
import datetime
import functools
import glob
import itertools
import json
import operator
import os
import re
import sqlite3
import sys
import time
import yaml

class TableRunsDetailsTupleClass(object):
    
    typename = 'TableRunsDetailsTuple'
    field_names = 'timestamp,hf,hl,seed,gpu,image_name,heigth,width,cropped_heigth,cropped_width,is_cropped,lr,epochs'.split(',')
    fields_type = 'str,int,int,int,str,str,int,int,int,int,str,float,int'.split(',')
    assert len(fields_type) == len(field_names), f'Error: len(field_type) != len(field_names) {len(fields_type)} != {len(field_names)}'
    fields_vals = [dict(type=eval(t), vals=v) for t,v in zip(fields_type, [None] * len(field_names))]
    TableRunsDetailsTuple = collections.namedtuple(typename, field_names)
    
    def __init__(self, *args, **kwargs):
        
        # pprint(args)
        # pprint(kwargs)
        
        self.a_record = TableRunsDetailsTupleClass.TableRunsDetailsTuple._make(TableRunsDetailsTupleClass.fields_vals)
        if kwargs is not None and len(kwargs.items()) != 0:
            for k, v in kwargs.items():
                self.__setitem__(k, v)
        elif args is not None and len(args) != 0:
            # print('kwargs is none or zero length')
            self.a_record = TableRunsDetailsTupleClass.TableRunsDetailsTuple._make(args)
            pass
        else:
            # print('kwargs is none and args is none or zero lengths')
            pass
        pass
    
    def __getitem__(self, key):
        return getattr(self.a_record, key)
    
    def __setitem__(self, key, value):
        # index = list(self.a_record._asdict().keys()).index(key)
        # self.a_record[index] = value
        tmp_dict = self.a_record._asdict()
        if isinstance(value, dict):
            tmp_dict[key] = value
        else:
            tmp_dict[key]['vals'] = str(value)
        self.a_record = TableRunsDetailsTupleClass.TableRunsDetailsTuple._make(tmp_dict.values())
        pass
    def __repr__(self,):
        return str(self.a_record._asdict())
    
    def get_to_insert(self,):
        def map_func(item):
            val_attr = item['vals']
            type_attr = item['type']
            if val_attr is None:
                return f"NULL"
            elif type_attr is str:
                val_attr = f"'{val_attr}'"
            else:
                val_attr = type_attr(val_attr)
                val_attr = f"{val_attr}"
                pass
            return f"{val_attr}"
        vals_str = '(' + ','.join([a_val for a_val in list(map(map_func, self.a_record._asdict().values()))]) + ')'
        keys_str = '(' + ','.join([a_key for a_key in self.a_record._asdict().keys()]) + ')'
        return keys_str, vals_str
                                  
                                   
                                   
    
    def get_as_constraint_repr(self,):
        def reduce_func(a, b):
            if a is None:
                return f"{b}"
            return f"{a} AND {b}"
        def map_func(item):
            pprint(item)
            atrr_name = item[0]
            val_attr = item[1]['vals']
            type_attr = item[1]['type']
            if val_attr is None:
                return f"{atrr_name} = NULL"
            elif type_attr is str:
                val_attr = f"'{val_attr}'"
            else:
                val_attr = type_attr(val_attr)
                val_attr = f"{val_attr}"
                pass
            return f"{atrr_name} = {val_attr}"
        return functools.reduce(reduce_func, map(map_func, self.a_record._asdict().items()), None)
    
    def keys(self):
        return self.a_record._asdict().keys()
    def values(self):
        return self.a_record._asdict().values()
    pass

class TableRunsDetailsClass(object):
        
    def __init__(self,):
        self.tmp_record = TableRunsDetailsTupleClass()
        self.records_list = []
        pass
    
    def get_default_record(self,):
        return copy.deepcopy(self.tmp_record)
    
    def make_record(self, vals):
        if isinstance(vals, list):
            return TableRunsDetailsTupleClass(*vals)
        elif isinstance(vals, dict):
            return TableRunsDetailsTupleClass(**vals)
        pass
    
    def append(self,a_record):
        self.records_list.append(copy.deepcopy(a_record))
        pass
    
    def update_record_by_dict(self, old_record, a_dict, relation_dict = None):
        record_updated = copy.deepcopy(old_record)
        if relation_dict is None:
            relation_dict = dict(zip(a_dict.keys(), old_record.keys()))
            pass
        for k,v in relation_dict.items():
            if v not in record_updated.keys(): continue
            record_updated[v]['vals'] = a_dict[k]
        return record_updated
    
    def get_all_records(self,):
        return copy.deepcopy(self.records_list)
    
    def get_all_records_processed_for_query(self,a_record):
        return list(map(operator.methodcaller('get_as_constraint_repr'), self.records_list))
    
    def get_all_records_processed_for_query_insert(self,):
        return list(map(operator.methodcaller('get_to_insert'), self.records_list))
    
    
    pass