# coding:utf-8
from calendar import c
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import os
import time
import sys
import datetime
import ctypes
import json
import numpy as np
import copy
from tqdm import tqdm


class WCGTrainerKuai16K(object):

    def __init__(
        self,
        model=None,
        data_loader=None,
        train_times=1000,
        alpha=0.5,
        use_gpu=True,
        opt_method="sgd",
        save_steps=None,
        checkpoint_dir=None,
        generator=None,
        lrg=None,
        mu=None,
        tester=None,
    ):

        self.work_threads = 8
        self.train_times = train_times

        self.opt_method = opt_method
        self.optimizer = None
        self.lr_decay = 0
        self.weight_decay = 0
        self.alpha = alpha
        # learning rate of the generator
        assert lrg is not None
        self.alpha_g = lrg

        self.model = model
        self.data_loader = data_loader
        self.use_gpu = use_gpu
        self.save_steps = 100
        self.checkpoint_dir = checkpoint_dir

        # the generator part
        assert generator is not None
        assert mu is not None
        self.optimizer_g = None
        self.generator = generator
        self.batch_size = self.model.batch_size
        self.generator.cuda()
        self.mu = mu
        self.tester = tester

    def train_one_step(self, data):
        # training D
        self.optimizer.zero_grad()
        loss, p_score = self.model(
            {
                "batch_h": self.to_var(data["batch_h"], self.use_gpu),
                "batch_t": self.to_var(data["batch_t"], self.use_gpu),
                "batch_r": self.to_var(data["batch_r"], self.use_gpu),
                "batch_y": self.to_var(data["batch_y"], self.use_gpu),
                "mode": data["mode"],
            }
        )
        # generate fake multimodal feature
        batch_h_gen = self.to_var(data["batch_h"][0 : self.batch_size], self.use_gpu)
        batch_t_gen = self.to_var(data["batch_t"][0 : self.batch_size], self.use_gpu)
        batch_r = self.to_var(data["batch_r"][0 : self.batch_size], self.use_gpu)
        batch_hs, batch_hi, batch_ht, batch_ha, batch_hv = (
            self.model.model.get_batch_ent_multimodal_embs(batch_h_gen)
        )
        batch_ts, batch_ti, batch_tt, batch_ta, batch_tv = (
            self.model.model.get_batch_ent_multimodal_embs(batch_t_gen)
        )
        batch_gen_hi, batch_gen_ht, batch_gen_ha, batch_gen_hv = self.generator(
            batch_hs, batch_hi, batch_ht, batch_ha, batch_hv
        )
        batch_gen_ti, batch_gen_tt, batch_gen_ta, batch_gen_tv = self.generator(
            batch_ts, batch_ti, batch_tt, batch_ta, batch_tv
        )
        scores, _ = self.model.model.get_fake_score(
            batch_h=batch_h_gen,
            batch_r=batch_r,
            batch_t=batch_t_gen,
            mode=data["mode"],
            fake_hi=batch_gen_hi,
            fake_ti=batch_gen_ti,
            fake_ht=batch_gen_ht,
            fake_tt=batch_gen_tt,
            fake_ha=batch_gen_ha,
            fake_ta=batch_gen_ta,
            fake_hv=batch_gen_hv,
            fake_tv=batch_gen_tv,
        )
        # when training D: positive_score > fake_score
        for score in scores:
            # print(p_score.shape, score.shape)
            loss += self.mu * (-torch.mean(p_score) + torch.mean(score))
        loss.backward()
        self.optimizer.step()
        # training G
        self.optimizer_g.zero_grad()
        batch_hs, batch_hi, batch_ht, batch_ha, batch_hv = (
            self.model.model.get_batch_ent_multimodal_embs(batch_h_gen)
        )
        batch_ts, batch_ti, batch_tt, batch_ta, batch_tv = (
            self.model.model.get_batch_ent_multimodal_embs(batch_t_gen)
        )
        batch_gen_hi, batch_gen_ht, batch_gen_ha, batch_gen_hv = self.generator(
            batch_hs, batch_hi, batch_ht, batch_ha, batch_hv
        )
        batch_gen_ti, batch_gen_tt, batch_gen_ta, batch_gen_tv = self.generator(
            batch_ts, batch_ti, batch_tt, batch_ta, batch_tv
        )
        scores, _ = self.model.model.get_fake_score(
            batch_h=batch_h_gen,
            batch_r=batch_r,
            batch_t=batch_t_gen,
            mode=data["mode"],
            fake_hi=batch_gen_hi,
            fake_ti=batch_gen_ti,
            fake_ht=batch_gen_ht,
            fake_tt=batch_gen_tt,
            fake_ha=batch_gen_ha,
            fake_ta=batch_gen_ta,
            fake_hv=batch_gen_hv,
            fake_tv=batch_gen_tv,
        )
        loss_g = 0.0
        for score in scores:
            loss_g += torch.mean(self.model.model.margin - score) / 3
        loss_g.backward()
        self.optimizer_g.step()
        return loss.item(), loss_g.item()

    def run(self):
        if self.use_gpu:
            self.model.cuda()

        if self.optimizer is not None:
            pass
        elif self.opt_method == "Adam" or self.opt_method == "adam":
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.alpha,
                weight_decay=self.weight_decay,
            )
            self.optimizer_g = optim.Adam(
                self.generator.parameters(),
                lr=self.alpha_g,
                weight_decay=self.weight_decay,
            )
            print(
                "Learning Rate of D: {}\nLearning Rate of G: {}".format(
                    self.alpha, self.alpha_g
                )
            )
        else:
            raise NotImplementedError
        print("Finish initializing...")

        for epoch in range(self.train_times):
            res = 0.0
            res_g = 0.0
            for data in self.data_loader:
                loss, loss_g = self.train_one_step(data)
                res += loss
                res_g += loss_g
            print("Epoch {} | D loss: {}, G loss {}".format(epoch, res, res_g))

            if self.save_steps and (epoch + 1) % self.save_steps == 0:
                print("Epoch %d has finished, validate..." % (epoch))
                self.tester.run_link_prediction(type_constrain=False)
                # self.model.save_checkpoint(os.path.join(self.checkpoint_dir + "-" + str(epoch) + ".ckpt"))

    def set_model(self, model):
        self.model = model

    def to_var(self, x, use_gpu):
        if use_gpu:
            return Variable(torch.from_numpy(x).cuda())
        else:
            return Variable(torch.from_numpy(x))

    def set_use_gpu(self, use_gpu):
        self.use_gpu = use_gpu

    def set_alpha(self, alpha):
        self.alpha = alpha

    def set_lr_decay(self, lr_decay):
        self.lr_decay = lr_decay

    def set_weight_decay(self, weight_decay):
        self.weight_decay = weight_decay

    def set_opt_method(self, opt_method):
        self.opt_method = opt_method

    def set_train_times(self, train_times):
        self.train_times = train_times

    def set_save_steps(self, save_steps, checkpoint_dir=None):
        self.save_steps = save_steps
        if not self.checkpoint_dir:
            self.set_checkpoint_dir(checkpoint_dir)

    def set_checkpoint_dir(self, checkpoint_dir):
        self.checkpoint_dir = checkpoint_dir
