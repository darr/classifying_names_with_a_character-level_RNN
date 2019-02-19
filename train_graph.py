#!/usr/bin/python
# -*- coding: utf-8 -*-
#####################################
# File name : train_graph.py
# Create date : 2019-02-01 17:22
# Modified date : 2019-02-19 13:06
# Author : DARREN
# Describe : not set
# Email : lzygzh@126.com
#####################################
from __future__ import division
from __future__ import print_function

import time
import status
from graph import RNNGraph

class TrainRNNGraph(RNNGraph):
    def __init__(self, data_dict, config):
        super(TrainRNNGraph, self).__init__(data_dict, config)

    def eval_a_epoch(self):
        loss, acc, corrects = self._eval_a_epoch()

    def train_a_epoch(self):
        loss, acc, corrects = self._train_a_epoch()

    def _run_a_epoch(self, epoch):
        status.update_epoch(epoch, self.status_dict)
        start = time.time()
        self.train_a_epoch()
        self.eval_a_epoch()
        end = time.time()

        status.update_elapsed_time(start, end, self.status_dict)
        status.save_epoch_status(self.status_dict, self.config)
        self._save_trained_model()

    def train_the_model(self):
        self._create_output()
        epoch = self.status_dict["epoch"]

        while True:
            if not self.check_epoch_stop():
                self._run_a_epoch(epoch)
                epoch += 1
            else:
                break

        self.show_the_value()
        return self.graph_dict["model"]
