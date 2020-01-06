##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: chenyuru
## This source code is licensed under the MIT-style license
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
import os
import sys
import time
import datetime
import numpy as np
import scipy.io
import shutil
from tensorboardX import SummaryWriter
from trainers import Trainer, DataPrefetcher
from utils import predict_multi_scale, predict_whole_img, compute_errors, \
                    display_figure, colored_depthmap, merge_images, measure_list
import torch
from torch.nn import DataParallel
import matplotlib.pyplot as plt
from tqdm import tqdm
from copy import deepcopy
import json

class DepthEstimationTrainer(Trainer):
    def __init__(self, params, net, datasets, criterion, optimizer, scheduler, 
                 sets=['train', 'val', 'test'], verbose=100, stat=False, 
                 eval_func=compute_errors, 
                 disp_func=display_figure):
        self.time = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        self.params = params
        self.verbose = verbose
        self.eval_func = eval_func
        self.disp_func = disp_func
        # Init dir
        if params.workdir is not None:
            workdir = os.path.expanduser(params.workdir)
        if params.logdir is None:
            logdir = os.path.join(workdir, 'log_{}_{}'.format(params.encoder+params.decoder, params.dataset))
        else:
            logdir = os.path.join(workdir, params.logdir)
        resdir = None
        if self.params.mode == 'test':
            if params.resdir is None:
                resdir = os.path.join(logdir, 'res')
            else:
                resdir = os.path.join(logdir, params.resdir)
        # Call the constructor of the parent class (Trainer)
        super().__init__(net, datasets, optimizer, scheduler, criterion,
                         batch_size=params.batch, batch_size_val=params.batch_val,
                         max_epochs=params.epochs, threads=params.threads, eval_freq=params.eval_freq,
                         use_gpu=params.gpu, resume=params.resume, mode=params.mode,
                         sets=sets, workdir=workdir, logdir=logdir, resdir=resdir)
        self.params.logdir = self.logdir
        self.params.resdir = self.resdir
        # params json
        if self.params.mode == 'train':
            with open(os.path.join(self.logdir, 'params_{}.json'.format(self.time)), 'w') as f:
                json.dump(vars(self.params), f)
        # uncomment to display the model complexity
        if stat:
            from torchstat import stat
            import tensorwatch as tw
            #net_copy = deepcopy(self.net)
            stat(self.net, (3, *self.datasets[sets[0]].input_size))
            exit()
            tw.draw_model(self.net, (1, 3, *self.datasets[sets[0]].input_size))
            #del net_copy
        self.print('###### Experiment Parameters ######')
        for k, v in vars(self.params).items():
            self.print('{0:<22s} : {1:}'.format(k, v))
        
    def train(self):
        torch.backends.cudnn.benchmark = True
        if self.logdir:
            self.writer = SummaryWriter(self.logdir)
        else:
            raise Exception("Log dir doesn't exist!")
        # Calculate total step
        self.n_train = len(self.trainset)
        self.steps_per_epoch = np.ceil(self.n_train / self.batch_size).astype(np.int32)
        self.verbose = min(self.verbose, self.steps_per_epoch)
        self.n_steps = self.max_epochs * self.steps_per_epoch
        self.print("{0:<22s} : {1:} ".format('trainset sample', self.n_train))
        # calculate model parameters memory
        para = sum([np.prod(list(p.size())) for p in self.net.parameters()])
        memory = para * 4 / (1024**2)
        self.print('Model {} : params: {:,}, Memory {:.3f}MB'.format(self.net._get_name(), para, memory))
        # GO!!!!!!!!!
        start_time = time.time()
        self.train_total_time = 0
        self.time_sofar = 0
        for epoch in range(self.start_epoch, self.max_epochs + 1):
            # Train one epoch
            total_loss = self.train_epoch(epoch)
            torch.cuda.empty_cache()
            # Decay Learning Rate
            if self.params.scheduler in ['step', 'plateau']:
                self.scheduler.step()
            # Evaluate the model
            if self.eval_freq and epoch % self.eval_freq == 0:
                measures = self.eval(epoch)
                torch.cuda.empty_cache()
                for k in sorted(list(measures.keys())):
                    self.writer.add_scalar(k, measures[k], epoch)
        self.print("Finished training! Best epoch {} best acc {:.4f}".format(self.best_epoch, self.best_acc))
        self.print("Spend time: {:.2f}h".format((time.time() - start_time) / 3600))
        net_type = type(self.net).__name__
        best_pkl = os.path.join(self.logdir, '{}_{:03d}.pkl'.format(net_type, self.best_epoch))
        modify = os.path.join(self.logdir, 'best.pkl')
        shutil.copyfile(best_pkl, modify)
        return

    def train_epoch(self, epoch):
        self.net.train()
        device = torch.device('cuda:0' if self.use_gpu else 'cpu')
        self.net.to(device)
        self.criterion.to(device)
        # Iterate over data.
        prefetcher = DataPrefetcher(self.trainloader)
        data = prefetcher.next()
        step = 0
        while data is not None:
            images, labels = data[0].to(device), data[1].to(device)
            before_op_time = time.time()
            self.optimizer.zero_grad()
            output = self.net(images)
            loss1, loss2, loss3, total_loss = self.criterion(output, labels, epoch)
            total_loss.backward()
            self.optimizer.step()
            fps = images.shape[0] / (time.time() - before_op_time)
            time_sofar = self.train_total_time / 3600
            time_left = (self.n_steps / self.global_step - 1.0) * time_sofar
            lr = self.optimizer.param_groups[0]['lr']
            if self.verbose > 0 and (step + 1) % (self.steps_per_epoch // self.verbose) == 0:
                print_str = 'Epoch[{:>2}/{:>2}] | Step[{:>4}/{:>4}] | fps {:4.2f} | Loss1 {:7.3f} | Loss2 {:7.3f} | Loss3 {:7.3f} | elapsed {:.2f}h | left {:.2f}h | lr {:.3e}'. \
                        format(epoch, self.max_epochs, step + 1, self.steps_per_epoch, fps, loss1, loss2, loss3, time_sofar, time_left, lr)
                if self.params.classifier == 'OHEM':
                    ratio = self.criterion.AppearanceLoss.ohem_ratio
                    print_str += ' | OHEM {:.4f}'.format(ratio)
                    self.writer.add_scalar('OHEM', ratio)
                self.print(print_str)
            self.writer.add_scalar('loss1', loss1, self.global_step)
            self.writer.add_scalar('loss2', loss2, self.global_step)
            self.writer.add_scalar('loss3', loss3, self.global_step)
            self.writer.add_scalar('total_loss', total_loss, self.global_step)
            self.writer.add_scalar('lr', lr, epoch)
            # Decay Learning Rate
            if self.params.scheduler == 'poly':
                self.scheduler.step()
            self.global_step += 1
            self.train_total_time += time.time() - before_op_time
            data = prefetcher.next()
            step += 1
        return total_loss

    def eval(self, epoch):
        torch.backends.cudnn.benchmark = True
        self.n_val = len(self.valset)
        self.print("{0:<22s} : {1:} ".format('valset sample', self.n_val))
        self.print("<-------------Evaluate the model-------------->")
        # Evaluate one epoch
        measures, fps = self.eval_epoch(epoch)
        acc = measures['a1']
        self.print('The {}th epoch, fps {:4.2f} | {}'.format(epoch, fps, measures))
        # Save the checkpoint
        self.save(epoch, acc)
        return measures

    def eval_epoch(self, epoch):
        device = torch.device('cuda:0' if self.use_gpu else 'cpu')
        self.net.to(device)
        self.criterion.to(device)
        self.net.eval()
        val_total_time = 0
        #measure_list = ['a1', 'a2', 'a3', 'rmse', 'rmse_log', 'log10', 'abs_rel', 'sq_rel']
        measures = {key: 0 for key in measure_list}
        with torch.no_grad():
            sys.stdout.flush()
            tbar = tqdm(self.valloader)
            rand = np.random.randint(len(self.valloader))
            for step, data in enumerate(tbar):
                images, labels = data[0].to(device), data[1].to(device)
                # forward
                before_op_time = time.time()
                y = self.net(images)
                depths = self.net.inference(y)
                duration = time.time() - before_op_time
                val_total_time += duration
                # accuracy
                new = self.eval_func(labels, depths)
                for k, v in new.items():
                    measures[k] += v.item()
                # display images
                if step == rand and self.disp_func is not None:
                    visuals = {'inputs': images, 'sim_map': y['sim_map'], 'labels': labels, 'depths': depths}
                    self.disp_func(self.writer, visuals, epoch)
                print_str = 'Test step [{}/{}].'.format(step + 1, len(self.valloader))
                tbar.set_description(print_str)
        fps = self.n_val / val_total_time
        measures = {key: round(value/self.n_val, 5) for key, value in measures.items()}
        return measures, fps

    def test(self):
        n_test = len(self.testset)
        device = torch.device('cuda:0' if self.use_gpu else 'cpu')
        self.net.to(device)
        self.net.eval()
        self.print("<-------------Test the model-------------->")
        colormaps = {'nyu': plt.cm.jet, 'kitti': plt.cm.plasma}
        cm = colormaps[self.params.dataset]
        #measure_list = ['a1', 'a2', 'a3', 'rmse', 'rmse_log', 'log10', 'abs_rel', 'sq_rel']
        measures = {key: 0 for key in measure_list}
        test_total_time = 0
        with torch.no_grad():
            #sys.stdout.flush()
            #tbar = tqdm(self.testloader)
            for step, data in enumerate(self.testloader):
                images, labels = data[0].to(device), data[1].to(device)
                before_op_time = time.time()
                scales = [1]
                if self.params.use_ms: 
                    scales = [1, 1.25]
                depths = predict_multi_scale(self.net, images, scales, self.params.classes, self.params.use_flip)
                duration = time.time() - before_op_time
                test_total_time += duration
                # accuracy
                new = self.eval_func(labels, depths)
                print_str = "Test step [{}/{}], a1: {:.5f}, rmse: {:.5f}.".format(step + 1, n_test, new['a1'], new['rmse'])
                self.print(print_str)
                #tbar.set_description(print_str)
                #sys.stdout.flush()
                images = images.cpu().numpy().squeeze().transpose(1, 2, 0)
                labels = labels.cpu().numpy().squeeze()
                depths = depths.cpu().numpy().squeeze()
                labels = colored_depthmap(labels, cmap=cm).squeeze()
                depths = colored_depthmap(depths, cmap=cm).squeeze()
                #fuse = merge_images(images, labels, depths, self.params.min_depth, self.params.max_depth)
                plt.imsave(os.path.join(self.resdir, '{:04}_rgb.png'.format(step)), images)
                plt.imsave(os.path.join(self.resdir, '{:04}_gt.png'.format(step)), labels)
                plt.imsave(os.path.join(self.resdir, '{:04}_depth.png'.format(step)), depths)
                for k, v in new.items():
                    measures[k] += v.item()
        fps = n_test / test_total_time
        measures = {key: round(value / n_test, 5) for key, value in measures.items()}
        self.print('Testing done, fps {:4.2f} | {}'.format(fps, measures))
        return


if __name__ == '__main__':
    pass
