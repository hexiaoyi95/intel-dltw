#!/usr/bin/env python
# encoding: utf-8

import importlib

def applications_factory(application_info):
    model = importlib.import_module(application_info)
    return model
