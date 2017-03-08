#!/usr/bin/env python
# -*- coding: utf-8 -*-  
# @author Zhang zhiming (zhangzhiming@)
# date

import re
import json
import sys
reload(sys)
sys.setdefaultencoding( "utf-8" )



stop_word_set = set()
for line in sys.stdin:
    line = line.strip()
    if len(line.strip()) == 0:
        continue
    stop_word_set.add(line)
stop_list = sorted(list(stop_word_set))
print '\n'.join(stop_list)
