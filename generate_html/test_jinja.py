#!/usr/bin/env python

import jinja2
import json
import sys
from jinja2 import Environment, PackageLoader
input_json = list()
for i in range(1,len(sys.argv)):
    with open(sys.argv[i],'r') as fp:
        input_json.append(json.load(fp))
env = Environment(loader=jinja2.FileSystemLoader('./'))
template = env.get_template('test_template.html')

ctxs = list()
title = list()
caption = list()
for aFile in input_json:
    aTitle = set()
    cases = aFile['cases_info']
    for case in cases:
        if case.has_key('report_path'):
            case.pop('report_path')
        aTitle.update(case.keys())
    aTitle.remove('test_result')
    aTitle = list(aTitle)
    aTitle.append('test_result')
    title.append(aTitle)
    ctxs.append(cases)
    caption.append('application:{}, cpu_type:{} '.format(aFile['application'],aFile['cpu_type']))

with open('test_result.html','w') as fp:
    #template.render(heads = ['head1','head2'])
    fp.write(template.render(tableNum=range(len(sys.argv)-1),caption = caption, ctxs=ctxs, title = title))
