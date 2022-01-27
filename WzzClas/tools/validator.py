import os
import time
import numpy as np
import logging
import torch
import torch.nn as nn

from collections import defaultdict
from tqdm import tqdm



device = torch.device("cuda")

class Pairs_Validator(object):
    def __init__(self, options, testloader, unknow_valloader):
        self.options = options
        self.testloader = testloader
        self.unknow_valloader = unknow_valloader

    def __call__(self, model, metric_fc, loss, epoch, options):
        # metrics initialization
        model.eval()
        metric_fc.eval()

        epoch_loss = 0
        epoch_acc = 0
        total_num = 0

        # test on closeset
        with torch.no_grad():
            for i, (img, label) in tqdm(enumerate(self.testloader), total=len(self.testloader)):
                batch_size = label.shape[0]
                # obtain data for training
                img = img.to(device)
                label = label.to(device)

                # output = model(img)
                embedding = model(img)
                embedding = embedding.view(embedding.size(0), -1)
                embedding = torch.nn.functional.normalize(embedding)

                output = metric_fc(embedding, label)
                batch_loss = loss(output, label)

                _, predicted = output.max(1)
                correct_samples = predicted.eq(label).sum().item()
                epoch_loss += batch_loss.item()
                epoch_acc += correct_samples
                total_num += len(label)

        print('Validation on Close Set,Loss: {}, Accuracy: {}'.format
              (epoch_loss / total_num,
               epoch_acc / total_num))


        unseen_rej = 0
        unseen_mis = 0
        unseen_total = 0

        # test on open set
        with torch.no_grad():
            for i, (img, label) in tqdm(enumerate(self.unknow_valloader), total=len(self.unknow_valloader)):
                batch_size = label.shape[0]
                # obtain data for training
                img = img.to(device)
                label = label.to(device)
                unseen_total += len(label)

                # output = model(img)
                embedding = model(img)
                embedding = embedding.view(embedding.size(0), -1)
                embedding = torch.nn.functional.normalize(embedding)

                output = metric_fc(embedding, label)
                output = torch.squeeze(output)
                scores = torch.nn.functional.softmax(output, dim=0).cpu()

                for pred_score in scores:
                    if max(pred_score) > options['threshold']:
                        unseen_mis += 1
                    else:
                        unseen_rej += 1

        print('Validation on Open Set,Reject Rate: {}, Miss Rate: {}'.format
              (unseen_rej / unseen_total,
               unseen_mis / unseen_total))

        # model_path = ""
        # # if it is validation when training, should save the models
        # if epoch > 0:
        #     # save checkpoint model
        #     model_path = os.path.join(
        #         self.options['save_dir'], '{}_{:.5f}.ckpt'.format(epoch, total_precision))
        #     state_dict = model.module.state_dict()
        #     for key in state_dict.keys():
        #         state_dict[key] = state_dict[key].cpu()
        #     torch.save({
        #         'state_dict': state_dict,
        #         'lr': optimizer.param_groups[0]['lr']},
        #         model_path)
        #     # update the last checkpoint
        #     with open(os.path.join(self.options['save_dir'], 'last_checkpoint'), 'w') as f:
        #         f.write('{}_{:.5f}'.format(epoch, total_precision))

        # return total_precision, average_precision, model_path
