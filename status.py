#!/usr/bin/python
# -*- coding: utf-8 -*-
#####################################
# File name : status.py
# Create date : 2019-02-01 13:41
# Modified date : 2019-02-19 12:35
# Author : DARREN
# Describe : not set
# Email : lzygzh@126.com
#####################################
from __future__ import division
from __future__ import print_function

import copy
import time
import math

import record

def get_status_dict():
    status_dict = {}
    status_dict["elapsed_time"] = 0.0
    status_dict["epoch"] = 0
    status_dict["step"] = 0
    status_dict["epoch_elapsed_time"] = 0.0
    status_dict["step_elapsed_time"] = 0.0
    status_dict["so_far_elapsed_time"] = 0.0

    status_dict["acc_str"] = ""
    status_dict["progress_str"] = ""


    status_dict["best_epoch"] = 0
    status_dict["best_epoch_acc"] = 0.0
    status_dict["best_epoch_loss"] = None
    status_dict["best_epoch_model_wts"] = None
    status_dict["train_epoch_loss"] = 0.0
    status_dict["train_epoch_acc"] = 0.0
    status_dict["eval_epoch_loss"] = 0.0
    status_dict["eval_epoch_acc"] = 0.0
    status_dict["train_epoch_loss_list"] = []
    status_dict["train_epoch_acc_list"] = []
    status_dict["eval_epoch_loss_list"] = []
    status_dict["eval_epoch_acc_list"] = []


    status_dict["best_step"] = 0
    status_dict["best_step_acc"] = 0.0
    status_dict["best_step_loss"] = None
    status_dict["best_step_model_wts"] = None
    status_dict["train_step_loss"] = 0.0
    status_dict["train_step_acc"] = 0.0
    status_dict["eval_step_loss"] = 0.0
    status_dict["eval_step_acc"] = 0.0
    status_dict["train_step_loss_list"] = []
    status_dict["train_step_acc_list"] = []
    status_dict["eval_step_loss_list"] = []
    status_dict["eval_step_acc_list"] = []

    return status_dict

def _append_record_list(data, data_key, list_name, cat_name, status_dict):
    dic = {}
    dic[cat_name] = status_dict[cat_name]
    dic[data_key] = data
    status_dict[list_name].append(dic)

def _update_data(category_name, data_key, data, mode, status_dict):
    status_key = "%s_%s_%s" % (mode, category_name, data_key)
    status_dict[status_key] = data
    _append_record_list(data, data_key, "%s_list" % status_key, category_name, status_dict)

def _update_eval_data(category_name, data_key, data, status_dict):
    mode = "eval"
    _update_data(category_name, data_key, data, mode, status_dict)

def _update_step_eval_data(eval_loss, eval_acc, status_dict):
    category_name = "step"
    _update_eval_data(category_name, "loss", eval_loss, status_dict)
    _update_eval_data(category_name, "acc", eval_acc, status_dict)

def _update_epoch_eval_data(eval_loss, eval_acc, status_dict):
    category_name = "epoch"
    _update_eval_data(category_name, "loss", eval_loss, status_dict)
    _update_eval_data(category_name, "acc", eval_acc, status_dict)

def _update_train_data(category_name, data_key, data, status_dict):
    mode = "train"
    _update_data(category_name, data_key, data, mode, status_dict)

def _update_step_train_data(eval_loss, eval_acc, status_dict):
    category_name = "step"
    _update_train_data(category_name, "loss", eval_loss, status_dict)
    _update_train_data(category_name, "acc", eval_acc, status_dict)

def _update_epoch_train_data(eval_loss, eval_acc, status_dict):
    category_name = "epoch"
    _update_train_data(category_name, "loss", eval_loss, status_dict)
    _update_train_data(category_name, "acc", eval_acc, status_dict)

def val_epoch_update_status_dict(loss, acc, model, status_dict):
    _update_epoch_eval_data(loss, acc, status_dict)

    if status_dict["best_epoch_loss"] is None or loss < status_dict["best_epoch_loss"]:
        status_dict["best_epoch"] = status_dict["epoch"]
        status_dict["best_epoch_loss"] = loss
        status_dict["best_epoch_acc"] = acc
        status_dict["best_epoch_model_wts"] = copy.deepcopy(model.state_dict())

def val_step_update_status_dict(loss, acc, model, status_dict):
    _update_step_eval_data(loss, acc, status_dict)

    if status_dict["best_step_loss"] is None or loss < status_dict["best_step_loss"]:
        status_dict["best_step"] = status_dict["step"]
        status_dict["best_step_loss"] = loss
        status_dict["best_step_acc"] = acc
        status_dict["best_step_model_wts"] = copy.deepcopy(model.state_dict())

def train_step_update_status_dict(loss, acc, status_dict):
    _update_step_train_data(loss, acc, status_dict)

def train_epoch_update_status_dict(loss, acc, status_dict):
    _update_epoch_train_data(loss, acc, status_dict)

def update_elapsed_time(start, end, status_dict):
    status_dict["epoch_elapsed_time"] = end - start
    status_dict["so_far_elapsed_time"] += status_dict["epoch_elapsed_time"]

def update_epoch(epoch, status_dict):
    status_dict["epoch"] = epoch

def update_step(step, status_dict):
    status_dict["step"] = step

def update_acc_str(acc_str, status_dict):
    status_dict["acc_str"] = acc_str

def update_progress_str(progress_str, status_dict):
    status_dict["progress_str"] = progress_str

def _get_epoch_str(status_dict, config):
    num_epochs = config["epochs"]
    epoch = status_dict["epoch"]
    if config["max_epoch_stop"]:
        epoch_str = "%s/%s" % (epoch, num_epochs)
    else:
        epoch_str = "%s" % epoch
    return epoch_str

def _get_step_str(status_dict, config):
    num_steps = config["steps"]
    step = status_dict["step"]
    if config["max_step_stop"]:
        step_str = "%s/%s" % (step, num_steps)
    else:
        step_str = "%s" % step
    return step_str

def _get_values(category_name, status_dict):
    train_loss = status_dict["train_%s_loss" % category_name]
    train_acc = status_dict["train_%s_acc" % category_name]
    eval_loss = status_dict["eval_%s_loss" % category_name]
    eval_acc = status_dict["eval_%s_acc" % category_name]

    return train_loss, train_acc, eval_loss, eval_acc

def _get_best_values_str(category_name, status_dict):
    best = status_dict["best_%s" % category_name]
    best_acc = status_dict["best_%s_acc" % category_name]
    best_loss = status_dict["best_%s_loss" % category_name]
    if best_loss is None:
        best_loss = 0.0
    return "%s Loss:%.6f Acc:%.6f" % (best, best_loss, best_acc)

def save_step_status(status_dict, config):
    category_name = "step"
    epoch_str = _get_epoch_str(status_dict, config)
    step_str = _get_step_str(status_dict, config)
    train_loss, train_acc, eval_loss, eval_acc = _get_values(category_name, status_dict)
    acc_str = status_dict["acc_str"]
    progress_str = status_dict["progress_str"]
    best_epoch_str = _get_best_values_str("epoch", status_dict)
    best_step_str = _get_best_values_str("step", status_dict)
    # pylint: disable=bad-continuation
    save_str = '[E:%s] [S:%s] [Train Loss:%.6f Acc:%.6f%s] [Val Loss:%.6f Acc:%.6f %s] [Best Epoch:%s] [Best Step:%s] Step status' % (
                            epoch_str,
                            step_str,
                            train_loss,
                            train_acc,
                            progress_str,
                            eval_loss,
                            eval_acc,
                            acc_str,
                            best_epoch_str,
                            best_step_str,
                            )

    # pylint: enable=bad-continuation
    record.save_content(config, save_str)

def save_epoch_status(status_dict, config):
    epoch_str = _get_epoch_str(status_dict, config)
    step_str = _get_step_str(status_dict, config)
    train_loss, train_acc, eval_loss, eval_acc = _get_values("epoch", status_dict)

    acc_str = status_dict["acc_str"]

    best_epoch_str = _get_best_values_str("epoch", status_dict)
    best_step_str = _get_best_values_str("step", status_dict)

    epoch_elapsed_time = status_dict["epoch_elapsed_time"]
    so_far_elapsed_time = status_dict["so_far_elapsed_time"]
    # pylint: disable=bad-continuation
    save_str = '[E:%s] [S:%s] [Train Loss:%.6f Acc:%.6f] [Val Loss:%.6f Acc:%.6f %s] [Best Epoch:%s] [Best Step:%s] [%.2fs %.1fs] Epoch status' % (
                            epoch_str,
                            step_str,
                            train_loss,
                            train_acc,
                            eval_loss,
                            eval_acc,
                            acc_str,
                            best_epoch_str,
                            best_step_str,
                            epoch_elapsed_time,
                            so_far_elapsed_time,
                            )

    # pylint: enable=bad-continuation
    record.save_content(config, save_str)

def time_since(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

#   def _get_progress_str(batch_idx, data, train_loader):
#       # pylint: disable=bad-continuation
#       progress_str = ' {}/{} ({:.0f}%)'.format(
#                           batch_idx * len(data),
#                           len(train_loader.dataset),
#                           100. * batch_idx / len(train_loader)
#                           )
#       # pylint: enable=bad-continuation
#       return progress_str

#   def _get_accuracy_str(correct, test_loader):
#       accuracy_str = '{}/{} ({:.0f}%)'.format(correct, len(test_loader.dataset), 100. * correct / len(test_loader.dataset))
#       return accuracy_str

def get_progress_str(batch_idx, train_loader):
    # pylint: disable=bad-continuation
    progress_str = ' {}/{} ({:.0f}%)'.format(
                        batch_idx,
                        train_loader,
                        100. * batch_idx / train_loader
                        )
    # pylint: enable=bad-continuation
    return progress_str

def get_accuracy_str(correct, test_loader):
    accuracy_str = '{}/{} ({:.0f}%)'.format(correct, test_loader, 100. * correct / test_loader)
    return accuracy_str
