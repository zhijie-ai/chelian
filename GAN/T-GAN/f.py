#!/usr/bin/python
# -*- coding: utf-8 -*-
# Copyright (c) Tencent Inc.
# Author: ranchodzu (ranchodzu@tencent.com)
# TIME: 2023/1/9 17:57
from collections import Counter

data = [ line.strip() for line in open('switch_Beep_at_low_speed.txt','r')]
dit = Counter(data)
print(dit)
