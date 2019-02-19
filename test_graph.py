#!/usr/bin/python
# -*- coding: utf-8 -*-
#####################################
# File name : test_graph.py
# Create date : 2019-02-01 17:21
# Modified date : 2019-02-19 13:07
# Author : DARREN
# Describe : not set
# Email : lzygzh@126.com
#####################################
from __future__ import division
from __future__ import print_function

import torch

import name_dataset
from graph import RNNGraph
import show
import record

def _normalize_confusion(n_categories, confusion):
    for i in range(n_categories):
        confusion[i] = confusion[i] / confusion[i].sum()
    return confusion

class TestRNNGraph(RNNGraph):
    def __init__(self, data_dict, config):
        super(TestRNNGraph, self).__init__(data_dict, config)
        self._load_train_model("test")

    def eval_and_show_confusion(self):
        data_dict = self.data_dict
        config = self.config
        category_lines = data_dict["category_lines"]
        all_categories = data_dict["all_categories"]
        n_categories = data_dict["n_categories"]

        confusion = torch.zeros(n_categories, n_categories)
        n_confusion = 10000

        for i in range(n_confusion):
            category, line, category_tensor, line_tensor = name_dataset.random_training_example(all_categories, category_lines, config)
            output = self.get_model_output(line_tensor)
            guess, guess_i = name_dataset.category_from_output(output, all_categories)
            category_i = all_categories.index(category)
            confusion[category_i][guess_i] += 1

        confusion = _normalize_confusion(n_categories, confusion)
        show.show_confusion(confusion, data_dict, config)

    def predict(self, input_line, rnn=None, n_predictions=3):
        if rnn is None:
            rnn = self.graph_dict["model"]
        all_categories = self.data_dict["all_categories"]
        config = self.config
        record.save_content(config, '\n> %s' % input_line)
        with torch.no_grad():
            output = self.get_model_output(name_dataset.line_to_tensor(input_line, config))
            record.save_content(config, output.numpy())
            #record.save_content(config, output.tolist())
            record.save_content(config, all_categories)
            topv, topi = output.topk(n_predictions, 1, True)
            predictions = []

            for i in range(n_predictions):
                value = topv[0][i].item()
                category_index = topi[0][i].item()
                record.save_content(config, '(%.2f) %s' % (value, all_categories[category_index]))
                predictions.append([value, all_categories[category_index]])

    def _test_a_epoch(self):
        self.predict('Dovesky')
        self.predict('Jackson')
        self.predict('Satoshi')
        self.predict('Foong')
        self.predict('Tsai')

    def _init_best_step_model(self):
        print("best step model")
        model = self.graph_dict["model"]
        model.load_state_dict(self.status_dict["best_step_model_wts"])

    def _init_best_epoch_model(self):
        print("best epoch model")
        model = self.graph_dict["model"]
        model.load_state_dict(self.status_dict["best_epoch_model_wts"])

    def _test_best_step_model(self):
        self._init_best_step_model()
        self._test_a_epoch()

    def _test_best_epoch_model(self):
        self._init_best_epoch_model()
        self._test_a_epoch()

    def test_the_model(self):
        self._test_best_step_model()
        self._test_best_epoch_model()
