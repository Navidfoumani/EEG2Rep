import os
import logging
import torch
import numpy as np
import pandas as pd
from collections import OrderedDict
import time
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from torch import nn
from itertools import product
import matplotlib.pyplot as plt
import torch.nn.functional as F
from eval import fit_lr, fit_svm, make_representation
from sklearn.metrics import accuracy_score, precision_recall_curve, roc_curve, auc
from Models.loss import l2_reg_loss
from Models import utils, analysis
from Models.optimizers import get_optimizer
import torch.distributed as dist
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import sklearn

logger = logging.getLogger('__main__')

NEG_METRICS = {'loss'}  # metrics for which "better" is less


class BaseTrainer(object):

    def __init__(self, model, pre_train_loader, train_loader, test_loader, config, optimizer=None, l2_reg=None, print_interval=10,
                 console=True, print_conf_mat=False):
        self.model = model
        self.pre_train_loader = pre_train_loader
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = config['device']
        self.optimizer = config['optimizer']
        self.loss_module = config['loss_module']
        self.l2_reg = l2_reg
        self.print_interval = print_interval
        self.printer = utils.Printer(console=console)
        self.print_conf_mat = print_conf_mat
        self.epoch_metrics = OrderedDict()
        self.save_path = config['output_dir']

    def train_epoch(self, epoch_num=None):
        raise NotImplementedError('Please override in child class')

    def evaluate(self, epoch_num=None, keep_all=True):
        raise NotImplementedError('Please override in child class')

    def print_callback(self, i_batch, metrics, prefix=''):
        total_batches = len(self.dataloader)
        template = "{:5.1f}% | batch: {:9d} of {:9d}"
        content = [100 * (i_batch / total_batches), i_batch, total_batches]
        for met_name, met_value in metrics.items():
            template += "\t|\t{}".format(met_name) + ": {:g}"
            content.append(met_value)

        dyn_string = template.format(*content)
        dyn_string = prefix + dyn_string
        self.printer.print(dyn_string)


class Self_Supervised_Trainer(BaseTrainer):
    def __init__(self, *args, **kwargs):

        super(Self_Supervised_Trainer, self).__init__(*args, **kwargs)
        if kwargs['print_conf_mat']:
            self.analyzer = analysis.Analyzer(print_conf_mat=True)
        self.mse = nn.MSELoss(reduction='none')
        self.gap = nn.AdaptiveAvgPool1d(1)

    def train_epoch(self, epoch_num=None):
        self.model.copy_weight()
        self.model = self.model.train()
        epoch_loss = 0  # total loss of epoch
        total_samples = 0  # total samples in epoch
        epoch_entropy_c = 0
        epoch_entropy_t = 0
        for i, batch in enumerate(self.pre_train_loader):
            X, targets, IDs = batch
            rep_mask, rep_mask_prediction, rep_contex, rep_target = self.model.pretrain_forward(X.to(self.device))

            # align_loss = self.mse(rep_mask, rep_mask_prediction).sum(dim=-1).sum().div(rep_mask.size(0))
            # align_loss = F.smooth_l1_loss(rep_mask, rep_mask_prediction)
            align_loss = F.mse_loss(rep_mask, rep_mask_prediction)
            # entropy_values_contex = batch_entropy(rep_contex)
            # entropy_values_target = batch_entropy(rep_target)
            # entropy_values_target = torch.std(rep_target[:, :, 5], dim=1).sum()

            '''
            x = self.gap(rep_mask.transpose(2, 1)).squeeze()
            y = self.gap(rep_mask_prediction.transpose(2, 1)).squeeze()

            x = x - x.mean(dim=0)
            y = y - y.mean(dim=0)

            std_x = torch.sqrt(x.var(dim=0) + 0.0001)
            std_y = torch.sqrt(y.var(dim=0) + 0.0001)
            std_loss = torch.mean(F.relu(1 - std_x)) / 2 + torch.mean(F.relu(1 - std_y)) / 2

            cov_x = (x.T @ x) / (len(targets) - 1)
            cov_y = (y.T @ y) / (len(targets) - 1)
            cov_loss = (off_diagonal(cov_x).pow_(2).sum().div(x.shape[-1])
                        + off_diagonal(cov_y).pow_(2).sum().div(x.shape[-1]))
            
            std_x = torch.sqrt(rep_mask.var(dim=0) + 0.0001)
            std_y = torch.sqrt(rep_mask_prediction.var(dim=0) + 0.0001)
            std_loss = torch.mean(F.relu(1 - std_x)) / 2 + torch.mean(F.relu(1 - std_y)) / 2

            # covariance loss
            x = rep_mask - rep_mask.mean(dim=0)
            y = rep_mask_prediction - rep_mask_prediction.mean(dim=0)

            cov_x = (x.transpose(1, 2) @ x) / (len(targets) - 1)
            cov_y = (y.transpose(1, 2) @ y) / (len(targets) - 1)
            cov_loss = (off_diagonal(cov_x).pow_(2).sum().div(x.shape[-1])
                        + off_diagonal(cov_y).pow_(2).sum().div(x.shape[-1]))
            
            '''
            y = self.gap(rep_mask_prediction.transpose(2, 1)).squeeze()
            y = y - y.mean(dim=0)

            std_y = torch.sqrt(y.var(dim=0) + 0.0001)
            std_loss = torch.mean(F.relu(1 - std_y))

            cov_y = (y.T @ y) / (len(targets) - 1)
            cov_loss = off_diagonal(cov_y).pow_(2).sum().div(y.shape[-1])
            # total_loss = align_loss + std_loss + (0.04 * cov_loss)
            total_loss = align_loss + std_loss + cov_loss

            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()
            self.model.momentum_update()

            total_samples += 1
            epoch_loss += total_loss.item()
            # epoch_entropy_c += entropy_values_contex.item()
            # epoch_entropy_t += entropy_values_target.item()
        epoch_loss = epoch_loss / total_samples  # average loss per sample for whole epoch
        # epoch_entropy_c = epoch_entropy_c / total_samples
        # epoch_entropy_t = epoch_entropy_t / total_samples
        self.epoch_metrics['epoch'] = epoch_num
        self.epoch_metrics['loss'] = epoch_loss
        self.epoch_metrics['align'] = align_loss
        self.epoch_metrics['std'] = std_loss
        self.epoch_metrics['cov'] = cov_loss
        if (epoch_num + 1) % 5 == 0:
            self.model.eval()
            train_repr, train_labels = make_representation(self.model, self.train_loader)
            test_repr, test_labels = make_representation(self.model, self.test_loader)
            clf = fit_lr(train_repr.cpu().detach().numpy(), train_labels.cpu().detach().numpy())
            y_hat = clf.predict(test_repr.cpu().detach().numpy())
            acc_test = accuracy_score(test_labels.cpu().detach().numpy(), y_hat)
            plot_tSNE(test_repr.cpu().detach().numpy(), test_labels.cpu().detach().numpy())
            print('Test_acc:', acc_test)
            result_file = open(self.save_path + '/linear_result.txt', 'a+')
            print('{0}, {1}, {2}, {3}, {4}'.format(int(epoch_num), acc_test, align_loss, std_loss, cov_loss),
                  file=result_file)
            result_file.close()

        return self.epoch_metrics, self.model


def plot_tSNE(data, labels):
    # Create a TSNE instance with 2 components (dimensions)
    tsne = TSNE(n_components=2, random_state=42)
    # Fit and transform the data using t-SNE
    embedded_data = tsne.fit_transform(data)

    # Separate data points for each class
    class_0_data = embedded_data[labels == 0]
    class_1_data = embedded_data[labels == 1]

    # Plot with plt.plot
    plt.figure(figsize=(6, 5))
    plt.plot(class_0_data[:, 0], class_0_data[:, 1], 'bo', label='Real')
    plt.plot(class_1_data[:, 0], class_1_data[:, 1], 'ro', label='Fake')
    plt.legend(fontsize='large')
    plt.grid(False)  # Remove grid
    plt.axis(False)
    plt.savefig('SSL.pdf', bbox_inches='tight', format='pdf')
    # plt.show()


def off_diagonal(x):
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


class FullGatherLayer(torch.autograd.Function):
    """
    Gather tensors from all process and support backward propagation
    for the gradients across processes.
    """

    @staticmethod
    def forward(ctx, x):
        output = [torch.zeros_like(x) for _ in range(dist.get_world_size())]
        dist.all_gather(output, x)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        all_gradients = torch.stack(grads)
        dist.all_reduce(all_gradients)
        return all_gradients[dist.get_rank()]


def batch_entropy(batch_representations):
    """
    Calculate the entropy among various samples in the batch.
    Args:
        batch_representations (torch.Tensor): 3D tensor of shape [batch_size, num_patches, representation_size].
    Returns:
        torch.Tensor: Entropy values for each sample in the batch, a 1D tensor of shape [batch_size].
    """
    # Calculate the probability distribution by applying softmax along the representation_size dimension
    batch_representations = torch.mean(batch_representations, dim=1)
    prob_distribution = F.softmax(batch_representations, dim=1)

    # Calculate the entropy using the softmax probabilities
    entropy = -torch.sum(prob_distribution * torch.log(prob_distribution + 1e-10), dim=1)

    # Mean the entropy across the representation patches to get the total entropy for each sample
    total_entropy = torch.mean(entropy)

    return total_entropy


def SS_train_runner(config, model, trainer, path):
    epochs = config['epochs']
    # epochs = 5
    optimizer = config['optimizer']
    loss_module = config['loss_module']
    start_epoch = 0
    total_start_time = time.time()
    tensorboard_writer = SummaryWriter('summary')
    metrics = []  # (for validation) list of lists: for each epoch, stores metrics like loss, ...
    save_best_model = utils.SaveBestModel()

    Total_loss = []
    for epoch in tqdm(range(start_epoch + 1, epochs + 1), desc='Training Epoch', leave=False):

        aggr_metrics_train, model = trainer.train_epoch(epoch)  # dictionary of aggregate epoch metrics
        # save_best_model(aggr_metrics_train['loss'], epoch, model, optimizer, loss_module, path)
        metrics_names, metrics_values = zip(*aggr_metrics_train.items())
        metrics.append(list(metrics_values))
        Total_loss.append(aggr_metrics_train['loss'])
        print_str = 'Epoch {} Training Summary: '.format(epoch)
        for k, v in aggr_metrics_train.items():
            tensorboard_writer.add_scalar('{}/train'.format(k), v, epoch)
            print_str += '{}: {:8f} | '.format(k, v)
        logger.info(print_str)
        # plot_loss(Total_loss,Time_loss,Freq_loss)
        if epoch > 50 or epochs < 50:
            save_best_model(aggr_metrics_train['loss'], epoch, model, optimizer, loss_module, path)
    total_runtime = time.time() - total_start_time
    logger.info("Train Time: {} hours, {} minutes, {} seconds\n".format(*utils.readable_time(total_runtime)))
    return


class SupervisedTrainer(BaseTrainer):

    def __init__(self, *args, **kwargs):
        super(SupervisedTrainer, self).__init__(*args, **kwargs)
        self.analyzer = analysis.Analyzer(print_conf_mat=False)
        if kwargs['print_conf_mat']:
            self.analyzer = analysis.Analyzer(print_conf_mat=True)

    def train_epoch(self, epoch_num=None):
        self.model = self.model.train()
        epoch_loss = 0  # total loss of epoch
        total_samples = 0  # total samples in epoch
        for i, batch in enumerate(self.train_loader):
            X, targets, IDs = batch
            targets = targets.to(self.device)
            predictions = self.model(X.to(self.device))
            loss = self.loss_module(predictions, targets)  # (batch_size,) loss for each sample in the batch
            batch_loss = torch.sum(loss)
            total_loss = batch_loss / len(loss)  # mean loss (over samples)

            # Zero gradients, perform a backward pass, and update the weights.
            self.optimizer.zero_grad()
            total_loss.backward()

            # torch.nn.utils.clip_grad_value_(self.model.parameters(), clip_value=1.0)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=4.0)
            self.optimizer.step()

            with torch.no_grad():
                total_samples += 1
                epoch_loss += total_loss.item()

        epoch_loss = epoch_loss / total_samples  # average loss per sample for whole epoch
        self.epoch_metrics['epoch'] = epoch_num
        self.epoch_metrics['loss'] = epoch_loss
        return self.epoch_metrics

    def evaluate(self, epoch_num=None, keep_all=True):

        self.model = self.model.eval()

        epoch_loss = 0  # total loss of epoch
        total_samples = 0  # total samples in epoch

        per_batch = {'targets': [], 'predictions': [], 'metrics': [], 'IDs': []}
        for i, batch in enumerate(self.train_loader):
            X, targets, IDs = batch
            targets = targets.to(self.device)
            predictions = self.model(X.to(self.device))
            loss = self.loss_module(predictions, targets)  # (batch_size,) loss for each sample in the batch
            batch_loss = torch.sum(loss).cpu().item()
            mean_loss = batch_loss / len(loss)  # mean loss (over samples)

            per_batch['targets'].append(targets.cpu().numpy())
            predictions = predictions.detach()
            per_batch['predictions'].append(predictions.cpu().numpy())
            loss = loss.detach()
            per_batch['metrics'].append([loss.cpu().numpy()])
            per_batch['IDs'].append(IDs)

            metrics = {"loss": mean_loss}
            # if i % self.print_interval == 0:
            # ending = "" if epoch_num is None else 'Epoch {} '.format(epoch_num)
            # self.print_callback(i, metrics, prefix='Evaluating ' + ending)

            total_samples += len(loss)
            epoch_loss += batch_loss  # add total loss of batch

        epoch_loss /= total_samples  # average loss per element for whole epoch
        self.epoch_metrics['epoch'] = epoch_num
        self.epoch_metrics['loss'] = epoch_loss

        predictions = torch.from_numpy(np.concatenate(per_batch['predictions'], axis=0))
        probs = torch.nn.functional.softmax(predictions,
                                            dim=1)  # (total_samples, num_classes) est. prob. for each class and sample
        predictions = torch.argmax(probs, dim=1).cpu().numpy()  # (total_samples,) int class index for each sample
        probs = probs.cpu().numpy()
        targets = np.concatenate(per_batch['targets'], axis=0).flatten()
        class_names = np.arange(probs.shape[1])  # TODO: temporary until I decide how to pass class names
        metrics_dict = self.analyzer.analyze_classification(predictions, targets, class_names)

        self.epoch_metrics['accuracy'] = metrics_dict['total_accuracy']  # same as average recall over all classes
        self.epoch_metrics['precision'] = metrics_dict['prec_avg']  # average precision over all classes

        if max(targets) < 2 == 2:
            false_pos_rate, true_pos_rate, _ = roc_curve(targets, probs[:, 1])  # 1D scores needed
            self.epoch_metrics['AUROC'] = auc(false_pos_rate, true_pos_rate)

            prec, rec, _ = precision_recall_curve(targets, probs[:, 1])
            self.epoch_metrics['AUPRC'] = auc(rec, prec)

        return self.epoch_metrics, metrics_dict


def validate(val_evaluator, tensorboard_writer, config, best_metrics, best_value, epoch):
    """Run an evaluation on the validation set while logging metrics, and handle outcome"""

    with torch.no_grad():
        aggr_metrics, ConfMat = val_evaluator.evaluate(epoch, keep_all=True)

    print()
    print_str = 'Validation Summary: '
    for k, v in aggr_metrics.items():
        tensorboard_writer.add_scalar('{}/val'.format(k), v, epoch)
        print_str += '{}: {:8f} | '.format(k, v)
    logger.info(print_str)

    if config['key_metric'] in NEG_METRICS:
        condition = (aggr_metrics[config['key_metric']] < best_value)
    else:
        condition = (aggr_metrics[config['key_metric']] > best_value)
    if condition:
        best_value = aggr_metrics[config['key_metric']]
        utils.save_model(os.path.join(config['save_dir'], 'model_best.pth'), epoch, val_evaluator.model)
        best_metrics = aggr_metrics.copy()

    return aggr_metrics, best_metrics, best_value


def Strain_runner(config, model, trainer, evaluator, path):
    epochs = config['epochs']
    # epochs = 100
    optimizer = config['optimizer']
    loss_module = config['loss_module']
    start_epoch = 0
    total_start_time = time.time()
    tensorboard_writer = SummaryWriter('summary')
    best_value = 1e16
    metrics = []  # (for validation) list of lists: for each epoch, stores metrics like loss, ...
    best_metrics = {}
    save_best_model = utils.SaveBestModel()
    # save_best_model = utils.SaveBestACCModel()
    for epoch in tqdm(range(start_epoch + 1, epochs + 1), desc='Training Epoch', leave=False):

        aggr_metrics_train = trainer.train_epoch(epoch)  # dictionary of aggregate epoch metrics
        aggr_metrics_val, best_metrics, best_value = validate(evaluator, tensorboard_writer, config, best_metrics,
                                                              best_value, epoch)
        save_best_model(aggr_metrics_val['loss'], epoch, model, optimizer, loss_module, path)
        # save_best_model(aggr_metrics_train['loss'], epoch, model, optimizer, loss_module, path)
        metrics_names, metrics_values = zip(*aggr_metrics_train.items())
        metrics.append(list(metrics_values))

        print_str = 'Epoch {} Training Summary: '.format(epoch)
        for k, v in aggr_metrics_train.items():
            tensorboard_writer.add_scalar('{}/train'.format(k), v, epoch)
            print_str += '{}: {:8f} | '.format(k, v)
        logger.info(print_str)
    total_runtime = time.time() - total_start_time
    logger.info("Train Time: {} hours, {} minutes, {} seconds\n".format(*utils.readable_time(total_runtime)))
    return
