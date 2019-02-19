#!/usr/bin/python
# -*- coding: utf-8 -*-
#####################################
# File name : main.py
# Create date : 2019-02-01 17:22
# Modified date : 2019-02-19 13:11
# Author : DARREN
# Describe : not set
# Email : lzygzh@126.com
#####################################
from __future__ import unicode_literals, print_function, division

from etc import config
import name_dataset
from train_graph import TrainRNNGraph
from test_graph import TestRNNGraph

def run():
    data_dict = name_dataset.get_data_dict(config)
    name_dataset.do_test(config)

    train_g = TrainRNNGraph(data_dict, config)
    train_g.train_the_model()

    test_g = TestRNNGraph(data_dict, config)
    test_g.eval_and_show_confusion()
    test_g.test_the_model()

run()
