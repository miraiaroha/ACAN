import os
import glob
import torch
import torch.nn as nn
import time
import datetime
import logging
import json
import numpy as np
from torch.utils.data import DataLoader

def init_log(output_dir, time, mode='train'):
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s-%(filename)s[l:%(lineno)d]: %(message)s',
                        datefmt='%Y%m%d-%H:%M:%S',
                        filename=os.path.join(output_dir, '{}_log_{}.log'.format(mode, time)),
                        filemode='a')
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logging.getLogger('').addHandler(console)
    return logging

class DataPrefetcher():
    """
       direct to: 
       https://github.com/NVIDIA/apex/blob/f5cd5ae937f168c763985f627bbf850648ea5f3f/examples/imagenet/main_amp.py#L256
    """
    def __init__(self, loader):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.preload()

    def preload(self):
        try:
            self.next_data = next(self.loader)
        except StopIteration:
            self.next_data = None
            return
        except TypeError:
            self.next_data = None
            return
        with torch.cuda.stream(self.stream):
            self.next_data = [x.cuda(non_blocking=True) for x in self.next_data]
            
    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        data = self.next_data
        self.preload()
        return data


class Trainer(object):
    """Base trainer class. Contains functions for training and saving/loading chackpoints.
    Trainer classes should inherit from this one and overload the train_epoch function."""

    def __init__(self, net, datasets, optimizer, scheduler, criterion,
                 batch_size, batch_size_val, max_epochs, threads, eval_freq,
                 use_gpu, resume, mode, sets, workdir, logdir, resdir):
        self.net = net
        self.datasets = datasets
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.batch_size = batch_size
        self.batch_size_val = batch_size_val
        self.max_epochs = max_epochs
        self.threads = threads
        self.eval_freq = eval_freq  # use 0 to disable test during training
        self.use_gpu = use_gpu
        self.resume = resume
        self.mode = mode
        self.sets = sets
        self.workdir = workdir
        self.logdir = logdir
        self.resdir = resdir
        # Init
        self.best_epoch = 0
        self.best_acc = 0
        self.start_epoch = 1
        self.global_step = 1
        #self.stats = {}
        # trainloader
        if 'train' in self.sets and self.datasets['train'] is not None:
            self.trainset = self.datasets[self.sets[0]]
            self.trainloader = DataLoader(self.trainset, self.batch_size,
                                      shuffle=True, num_workers=self.threads, 
                                      pin_memory=True, drop_last=False,
                                      worker_init_fn=lambda work_id:np.random.seed(work_id))
        elif self.params.mode != 'test':
            raise Exception("train set not found!")
        # valloader
        if 'train' in self.sets and self.datasets['val'] is not None:        
            self.valset = self.datasets[self.sets[1]]
            self.valloader = DataLoader(self.valset, self.batch_size_val,
                                      shuffle=False, num_workers=self.threads, 
                                      pin_memory=Trainer, drop_last=False,
                                      worker_init_fn=lambda work_id:np.random.seed(work_id))
        elif self.params.mode != 'test':
            raise Exception("Val set not found!")
        # testloader
        if 'test' in self.sets and self.datasets['test'] is not None:
            self.testset = self.datasets[self.sets[2]]
            self.testloader = DataLoader(self.testset, 1, shuffle=False)
        # Workspace dir
        if self.workdir is not None:
            if not os.path.exists(self.workdir):
                os.makedirs(self.workdir)
        else:
            raise Exception("Workspace directory doesn't exist!")
        # log dir
        if self.logdir is not None:
            if not os.path.exists(self.logdir):
                os.mkdir(self.logdir)
            logging = init_log(self.logdir, self.time, self.params.mode)
            self.print = logging.info
        else:
            self.print = print
        # result dir
        if self.resdir is not None:
            self.resdir = '{}_{}'.format(self.resdir, self.time)
            if not os.path.exists(self.resdir):
                os.makedirs(self.resdir)

    def train(self):
        """Do training, you can overload this function according to your need."""
        self.print("Log dir: {}".format(self.logdir))
        # Calculate total step
        self.n_train = len(self.trainset)
        self.steps_per_epoch = np.ceil(
            self.n_train / self.batch_size).astype(np.int32)
        self.verbose = min(self.verbose, self.steps_per_epoch)
        self.n_steps = self.max_epochs * self.steps_per_epoch
        # calculate model parameters memory
        para = sum([np.prod(list(p.size())) for p in self.net.parameters()])
        memory = para * 4 / 1000 / 1000
        self.print('Model {} : params: {:4f}M'.format(self.net._get_name(), memory))
        self.print('###### Experiment Parameters ######')
        for k, v in self.params.items():
            self.print('{0:<22s} : {1:}'.format(k, v))
        self.print("{0:<22s} : {1:} ".format('trainset sample', self.n_train))
        # GO!!!!!!!!!
        start_time = time.time()
        self.train_total_time = 0
        self.time_sofar = 0
        for epoch in range(self.start_epoch, self.max_epochs + 1):
            # Decay Learning Rate
            self.scheduler.step()
            # Train one epoch
            total_loss = self.train_epoch(epoch)
            torch.cuda.empty_cache()
            # Evaluate the model
            if self.eval_freq and epoch % self.eval_freq == 0:
                acc = self.eval(epoch)
                torch.cuda.empty_cache()
        self.print("Finished training! Best epoch {} best acc {}".format(self.best_epoch, self.best_acc))
        self.print("Spend time: {:.2f}h".format((time.time() - start_time) / 3600))

    def train_epoch(self, epoch):
        """Train one epoch, you can overload this function according to your need."""
        device = torch.device('cuda:0' if self.use_gpu else 'cpu')
        self.net.to(device)
        self.criterion.to(device)
        self.net.train()
        # Iterate over data.
        prefetcher = DataPrefetcher(self.trainloader)
        image, label = prefetcher.next()
        step = 0
        while image is not None:
            image, label = data[0].to(device), data[1].to(device)
            before_op_time = time.time()
            self.optimizer.zero_grad()
            output = self.net(image)
            total_loss = self.criterion(output, label)
            total_loss.backward()
            self.optimizer.step()
            fps = image.shape[0] / (time.time() - before_op_time)
            time_sofar = self.train_total_time / 3600
            time_left = (self.n_steps / self.global_step - 1.0) * time_sofar
            if self.verbose > 0 and (step + 1) % (self.steps_per_epoch // self.verbose) == 0:
                print_str = 'Epoch [{:>3}/{:>3}] | Step [{:>3}/{:>3}] | fps {:4.2f} | Loss: {:7.3f} | Time elapsed {:.2f}h | Time left {:.2f}h'. \
                    format(epoch, self.max_epochs, step + 1, self.steps_per_epoch,
                           fps, total_loss, time_sofar, time_left)
                self.print(print_str)
            self.global_step += 1
            self.train_total_time += time.time() - before_op_time
        image, label = prefetcher.next()
        step += 1
        return total_loss

    def eval(self, epoch):
        """Do evaluating, you can overload this function according to your need."""
        torch.backends.cudnn.benchmark = True
        self.n_val = len(self.valset)
        self.print("{0:<22s} : {1:} ".format('valset sample', self.n_val))
        self.print("<-------------Evaluate the model-------------->")
        # Evaluate one epoch
        acc = self.eval_epoch()
        self.print('The {}th epoch | acc: {:.4f}%'.format(epoch, acc * 100))
        # Save the checkpoint
        if self.logdir:
            self.save_checkpoint(epoch, acc)
            self.print('=> Checkpoint was saved successfully!')
        else:
            if acc >= self.best_acc:
                self.best_epoch, self.acc = epoch, acc
        return acc

    def eval_epoch(self):
        """Evaluate one epoch, you can overload this function according to your need."""
        if self.eval_freq:
            raise NotImplementedError

    def save(self, epoch, acc):
        self._save_checkpoint(epoch, acc)
        self.print('=> Checkpoint was saved successfully!')
        self._update_best_epoch(epoch, acc)

    def _update_best_epoch(self, epoch, acc):
        net_type = type(self.net).__name__
        if acc >= self.best_acc:
            last_best = os.path.join(self.logdir, '{}_{:03d}.pkl'.format(net_type, self.best_epoch))
            if os.path.isfile(last_best):
                os.remove(last_best)
            self.best_epoch = epoch
            self.best_acc = acc

    def _save_checkpoint(self, epoch, acc):
        """
        Saves a checkpoint of the network and other variables.
        Only save the best and latest epoch.
        """
        net_type = type(self.net).__name__
        if epoch - self.eval_freq != self.best_epoch:
            pre_save = os.path.join(self.logdir, '{}_{:03d}.pkl'.format(net_type, epoch - self.eval_freq))
            if os.path.isfile(pre_save):
                os.remove(pre_save)
        cur_save = os.path.join(self.logdir, '{}_{:03d}.pkl'.format(net_type, epoch))
        state = {
            'epoch': epoch,
            'acc': acc,
            'net_type': net_type,
            'net': self.net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            #'scheduler': self.scheduler.state_dict(),
            'use_gpu': self.use_gpu,
            'save_time': datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        }
        torch.save(state, cur_save)
        return True

    def reload(self, resume, mode='finetune'):
        if resume:
            if self._load_checkpoint(resume, mode):
                self.print('=> Checkpoint was loaded successfully!')
        else:
            raise Exception("Resume doesn't exist!")

    def _load_checkpoint(self, checkpoint=None, mode='finetune'):
        """
        Parameters
        ----------
        checkpoint:
            Loads a network checkpoint file. Can be called in three different ways:
                _load_checkpoint(epoch_num):
                    Loads the network at the given epoch number (int).
                _load_checkpoint(path_to_checkpoint):
                    Loads the file from the given absolute path (str).
                _load_checkpoint(net):
                    Loads the parameters from the given net (nn.Module).
        mode:
            'fintune': load the pretrained model and start training from epoch 1,
            'retain': reload the model and restart training from the breakpoint. 
            'test': just testing the model.
        """
        net_type = type(self.net).__name__
        if isinstance(checkpoint, nn.Module):
            old_model_dict = checkpoint.state_dict()
        else:
            if isinstance(checkpoint, int):
                # Checkpoint is the epoch number
                if self.logdir is not None:
                    checkpoint_path = os.path.join(self.logdir, '{}_{:03d}.pkl'.format(net_type, checkpoint))
                else:
                    raise Exception("Log dir doesn't exist!")
            elif isinstance(checkpoint, str):
                # checkpoint is the path
                checkpoint_path = os.path.expanduser(checkpoint)
            else:
                raise TypeError
            checkpoint_dict = torch.load(checkpoint_path, map_location={'cuda:1': 'cuda:0'})
            old_model_dict = checkpoint_dict['net']
        #assert net_type == checkpoint_dict['net_type'], 'Network is not of correct type.'
        if mode == 'finetune':
            net_state_dict = self.net.state_dict()
            new_model_dict = {key: value for key, value in old_model_dict.items()
                              if key in net_state_dict and value.shape == net_state_dict[key].shape}
            net_state_dict.update(new_model_dict)
            start_epoch = 1
            best_acc = 0
            use_gpu = self.use_gpu
        elif mode == 'retain':
            net_state_dict = checkpoint_dict['net']
            start_epoch = checkpoint_dict['epoch'] + 1
            best_acc = checkpoint_dict['acc']
            self.optimizer.load_state_dict(checkpoint_dict['optimizer'])
            if 'scheduler' in checkpoint_dict:
                self.scheduler.load_state_dict(checkpoint_dict['scheduler'])
                self.scheduler.last_epoch = start_epoch - 1
            use_gpu = checkpoint_dict['use_gpu']
        elif mode == 'test':
            net_state_dict = checkpoint_dict['net']
            use_gpu = checkpoint_dict['use_gpu']
            start_epoch = checkpoint_dict['epoch']
            best_acc = checkpoint_dict['acc']
        self.start_epoch = start_epoch
        self.best_acc = best_acc
        self.net.load_state_dict(net_state_dict)
        self.use_gpu = use_gpu
        return True


if __name__ == '__main__':
    pass
