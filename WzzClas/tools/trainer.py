import numpy as np
import logging
import torch
import time

from zzClassifier.datasets import mixup_data

device = torch.device("cuda")

class Pairs_Trainer():
    def __init__(self, options, data_loader, unknow_loader):
        self.options = options
        self.data_loader = data_loader
        self.unknow_loader = unknow_loader

    def train_epoch(self, model, criterion, criterion_tml, metric_fc, optimizer, loss, epoch, options):
        epoch_loss = 0
        epoch_acc = 0
        total_num = 0
        # training begaining
        model = model.to(device)
        model.train()

        metric_fc = metric_fc.to(device)
        metric_fc.train()

        start_time = time.time()
        total_batch_time = 0
        total_batch = len(self.data_loader)

        for step, (img, label) in enumerate(self.data_loader, start=1):
            batch_start = time.time()
            img = img.to(device)
            label = label.to(device)
            total_num += len(label)

            hard_negative_inputs, _ = iter(self.unknow_loader).next()
            # positive_inputs, _ = iter(self.background_loader).next()

            # # Hard Positive Example Sampers for training backbone
            # hard_positive_inputs, targets_a, targets_b, lam = mixup_data(img, positive_inputs, 1.0, True)
            # hard_positive_inputs = hard_positive_inputs.to(device)

            # Hard Negative Example Sampers
            hard_negative_inputs = hard_negative_inputs.to(device)


            # train one step
            output, losses = self.train_step(model, metric_fc, optimizer, loss, img, hard_negative_inputs, label)
            _, predicted = output.max(1)
            correct_samples = predicted.eq(label).sum().item()
            epoch_loss += losses.item()
            epoch_acc += correct_samples

        # end of this epoch
        end_time = time.time()
        print('Epoch: {},Loss: {}, Accuracy: {},train_time: {}'.format
              (epoch,
               epoch_loss / total_num,
               epoch_acc / total_num,
               end_time - start_time))

        return model, metric_fc

    def train_step(self, model, criterion, criterion_tml, metric_fc, optimizer, loss, img, hard_negative_inputs, label):
        # forward
        # embedding = model(img)
        # embedding = embedding.view(embedding.size(0), -1)
        # embedding = torch.nn.functional.normalize(embedding)
        #
        # output = metric_fc(embedding, label)
        # batch_loss = loss(output, label)
        output, global_feat, feat = model(img)
        celoss = criterion(output, label)

        hard_negative_inputs_output, hard_negative_inputs_global_feat, hard_negative_inputs_feat = model(
            hard_negative_inputs)
        tmloss = criterion_tml(global_feat, hard_positive_inputs_global_feat, hard_negative_inputs_global_feat)
        losses = celoss + tmloss

        # backward
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        return output, losses

    def train_step_combine(self, model, optimizer, loss, img, label):
        # forward
        output = model(img)
        output = output.view(output.size(0), -1)
        output = torch.nn.functional.normalize(output)

        batch_loss = loss(output, label)
        losses = batch_loss

        # backward
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        return output, losses

class Normal_Trainer():
    def __init__(self, options, data_loader):
        pass
    def train_epoch(self):
        pass
    def train_step(self):
        pass

class Gan_Trainer():
    def __init__(self, options, data_loader):
        pass
    def train_epoch(self):
        pass
    def train_step(self):
        pass

