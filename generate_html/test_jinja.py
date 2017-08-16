#!/usr/bin/env python

import jinja2
import json
import sys
import os
from jinja2 import Environment, PackageLoader
SCRIPT_HOME=os.path.dirname(os.path.realpath(__file__))
input_json = list()
for i in range(1,len(sys.argv)):
    with open(sys.argv[i],'r') as fp:
        input_json.append(json.load(fp))
env = Environment(loader=jinja2.FileSystemLoader(SCRIPT_HOME))
template = env.get_template(os.path.join('test_template.html'))

ctxs = list()
title = list()
caption = list()
            
for aFile in input_json:
    aTitle = set()
    cases = [ i for i in aFile['cases_info'] if not i['is_ref']]
    for index,case in enumerate(cases):
        if case.has_key('report_path'):
            case.pop('report_path')
	if case.has_key('is_ref'):
	    case.pop('is_ref')
        aTitle.update(case.keys())
    aTitle = list(aTitle)
    
    if 'test_result' in aTitle:
        aTitle.remove('test_result')
        aTitle.append('test_result')
    if 'topology' in aTitle:
        aTitle.remove('topology')
        aTitle.insert(0,'topology')
    aCaption='application: {}, cpu_type: {} '.format(aFile['application'],aFile['cpu_type'])
    if aCaption not in caption:
    	caption.append(aCaption)
    	ctxs.append(cases)
    	title.append(aTitle)
    else:
	index = caption.index(aCaption)
	ctxs[index].extend(cases)

with open(os.path.join(SCRIPT_HOME,'test_result.html'),'w') as fp:
    fp.write(template.render(tableNum=range(len(sys.argv)-1),caption = caption, ctxs=ctxs, title = title))
