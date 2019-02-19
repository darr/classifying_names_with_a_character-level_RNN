#!/usr/bin/python
# -*- coding: utf-8 -*-
#####################################
# File name : graph.py
# Create date : 2019-01-30 14:25
# Modified date : 2019-02-19 13:02
# Author : DARREN
# Describe : not set
# Email : lzygzh@126.com
#####################################
from __future__ import division
from __future__ import print_function

import torch.nn as nn
import torch.optim as optim

from rnn_model import RNN
from base_graph import BaseGraph
import name_dataset
import status

def _get_criterion():
    return nn.NLLLoss()

def _get_sgd_optimizer(model, config):
    learn_rate = config["learn_rate"]
    optimizer = optim.SGD(model.parameters(), lr=learn_rate)
    return optimizer

def _get_momentum_optimizer(model, config):
    learn_rate = config["learn_rate"]
    momentum = config["momentum"]
    optimizer = optim.SGD(model.parameters(), lr=learn_rate, momentum=momentum)
    return optimizer

def _get_model(data_dict, config):
    n_hidden = config["n_hidden"]
    n_letters = config["n_letters"]
    n_categories = data_dict["n_categories"]
    model = RNN(n_letters, n_hidden, n_categories).to(config["device"])
    return model

class RNNGraph(BaseGraph):
    def __init__(self, data_dict, config):
        super(RNNGraph, self).__init__(data_dict, config)

    def _deal_a_step(self, category_tensor, line_tensor, mode="test"):
        criterion = self.graph_dict["criterion"]
        rnn = self.graph_dict["model"]
        optimizer = self.graph_dict["optimizer"]

        if mode == "train":
            rnn.zero_grad()
            optimizer.zero_grad()

        output = self.get_model_output(line_tensor)
        loss = criterion(output, category_tensor)

        if mode == "train":
            loss.backward()
            optimizer.step()

        return output, loss.item()

    def _train_a_step(self, category_tensor, line_tensor):
        return self._deal_a_step(category_tensor, line_tensor, "train")

    def _eval_a_step(self, category_tensor, line_tensor):
        return self._deal_a_step(category_tensor, line_tensor, "test")

    def _init_graph_dict(self, config):
        graph_dict = {}
        graph_dict["model"] = _get_model(self.data_dict, config)

        if config["loss"] == "NLL":
            graph_dict["criterion"] = _get_criterion()

        if config["optimizer"] == "momentum":
            graph_dict["optimizer"] = _get_momentum_optimizer(graph_dict["model"], config)

        if config["optimizer"] == "SGD":
            graph_dict["optimizer"] = _get_sgd_optimizer(graph_dict["model"], config)

        return graph_dict

    def get_model_output(self, line_tensor):
        model = self.graph_dict["model"]
        hidden = model.init_hidden()
        hidden = hidden.to(self.config["device"])
        for i in range(line_tensor.size()[0]):
            ipt = line_tensor[i]
            output, hidden = model(ipt, hidden)
        return output

    def _eval_a_batch(self):
        return self._deal_a_batch(self._eval_a_step)

    def _train_a_batch(self):
        return self._deal_a_batch(self._train_a_step)

    def _deal_a_batch(self, step_func):
        data_dict = self.data_dict
        config = self.config
        batch_size = self.config["batch_size"]
        category_lines = data_dict["category_lines"]
        all_categories = data_dict["all_categories"]

        running_loss = 0.0
        running_corrects = 0

        for i in range(0, batch_size):
            category, line, category_tensor, line_tensor = name_dataset.random_training_example(all_categories, category_lines, config)
            line_tensor = line_tensor.to(config["device"])
            category_tensor = category_tensor.to(config["device"])
            output, loss = step_func(category_tensor, line_tensor)
            predict_category, category_index = name_dataset.category_from_output(output, all_categories)

            running_loss += loss
            if predict_category == category:
                running_corrects += 1

        avg_acc = running_corrects / batch_size
        avg_loss = running_loss / batch_size
        return avg_loss, avg_acc, running_corrects

    def _update_train_step_status(self, step, loss, acc, i, n_iters):
        model = self.graph_dict["model"]
        status_dict = self.status_dict
        status.update_step(step, self.status_dict)
        status.train_step_update_status_dict(loss, acc, self.status_dict)

        progress_str = status.get_progress_str(i, n_iters)
        status.update_progress_str(progress_str, self.status_dict)

        eval_loss, eval_acc, eval_corrects = self._eval_a_epoch()
        status.val_step_update_status_dict(eval_loss, eval_acc, model, status_dict)

        status.save_step_status(self.status_dict, self.config)

    def _deal_a_epoch(self, batch_func, mode="test"):
        model = self.graph_dict["model"]
        start_step = self.status_dict["step"]
        config = self.config
        running_loss = 0.0
        running_acc = 0.0
        running_corrects = 0

        if mode == "train":
            model.train()
            step = start_step
        else:
            model.eval()

        if mode == "train":
            n_iters = config["train_epoch_steps"]
        else:
            n_iters = config["eval_epoch_steps"]

        for i in range(1, n_iters + 1):
            loss, acc, corrects = batch_func()
            running_loss += loss
            running_acc += acc
            running_corrects += corrects

            if mode == "train":
                step += 1
                if step % config["print_every"] == 0:
                    self._update_train_step_status(step, loss, acc, i, n_iters)
                    if self.check_step_stop():
                        break

        epoch_loss = running_loss / n_iters
        epoch_acc = running_acc / n_iters

        if mode == "train":
            status.train_epoch_update_status_dict(epoch_loss, epoch_acc, self.status_dict)
        else:
            accuracy_str = status.get_accuracy_str(running_corrects, config["eval_epoch_steps"] * config["batch_size"])
            status.update_acc_str(accuracy_str, self.status_dict)
            status.val_epoch_update_status_dict(epoch_loss, epoch_acc, model, self.status_dict)

        return epoch_loss, epoch_acc, running_corrects

    def _eval_a_epoch(self):
        return self._deal_a_epoch(self._eval_a_batch, "test")

    def _train_a_epoch(self):
        return self._deal_a_epoch(self._train_a_batch, "train")
