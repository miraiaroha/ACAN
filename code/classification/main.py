import numpy as np
import time
import os
import sys
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision.models as models
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.autograd import Variable
sys.path.append('./dataset')
from dataset.nyu_dataloader import NyuFromTxt
from dataset.kitti_dataloader import KittiFromTxt
sys.path.append('./model')
from model.context_attention_model import ContextAttentionModel
from model.modules import *
from utils import *
from tensorboardX import SummaryWriter
import torchvision
from PIL import Image

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['GPU_DEBUG'] = '0'

def write_to_log(path, str):
    with open(path, 'a+') as f:
        f.write(str + '\n')

def generate_all_params(modules):
    generator_ = []
    for module in modules:
        list_ = list(module.parameters())
        generator_ = generator_ + list_
    generator = (i for i in generator_)
    return generator

def generate_weight_or_bias_params(modules, Type='weight', mode='conv_and_fc'):
    # mode
    # conv_and_fc
    # bnall
    # 
    generator_ = []
    for module in modules:
        params_dict = dict(module.named_parameters())
        for name, params in params_dict.items():
            if mode == 'bn':
                if 'bn' in name and Type in name:
                    generator_ += [params]
            elif mode == 'all':
                if Type in name:
                    generator_ += [params]
            else:
                if 'bn' not in name and Type in name:
                    generator_ += [params]
    generator = (i for i in generator_)
    return generator

def fine_tune(Pretrained, myModel):
    my_model_dict = myModel.state_dict()
    new_model_dict = {key: value for key, value in Pretrained.state_dict(
    ).items() if key in my_model_dict}
    my_model_dict.update(new_model_dict)
    myModel.load_state_dict(my_model_dict)
    return myModel

def adjust_learning_rate(optimizer, step, total_step, epoch, num_epochs, mode='step'):
    # step  [1,total_step]
    # epoch [1, num_epochs]
    if mode == 'lr_finder':
        start_lr = 1e-8   # 6e-5
        end_lr = 1e-3     # 1e-4
        new_lr = (end_lr - start_lr) / (num_epochs - 1) * \
            (epoch - 1) + start_lr
    if mode == 'leslie':
        a1 = 0.4 * num_epochs
        a2 = 2 * a1
        low_lr = 1e-8  # 1e-8 Adam
        high_lr = 1e-4  # 1e-5 Adam
        if epoch <= a1:
            new_lr = (high_lr - low_lr) / (a1 - 1) * (epoch - 1) + low_lr
        elif epoch <= a2:
            new_lr = (low_lr - high_lr) / (a2 - a1) * (epoch - a1) + high_lr
        else:
            new_lr = low_lr
    if mode == 'poly':
        start_lr = 1e-4
        decay_rate = 0.9
        new_lr = start_lr * (1 - (step - 1) / total_step)**decay_rate
    if mode == 'step':
        start_lr = 1e-4
        start_epoch = 2
        divider = 0.1
        interval = 1
        if epoch < start_epoch:
            new_lr = start_lr
        else:
            new_lr = start_lr * \
                (divider**((epoch - start_epoch) // interval + 1))
    if mode == 'constant':
        new_lr = 1e-5
    optimizer.param_groups[0]['lr'] = new_lr
    optimizer.param_groups[1]['lr'] = new_lr * 10
    optimizer.param_groups[2]['lr'] = new_lr * 20
    return new_lr
# ----------------------------------------------------------------------------

def main():
    mode = 'train'
    test_index = 13
    dataset_name_list = ['nyu', 'kitti']
    model_name_list = ['contextattention']
    dataset_dict = {'nyu': NyuFromTxt, 'kitti': KittiFromTxt}
    model_dict = {'contextattention': ContextAttentionModel}
    dataset_name = dataset_name_list[0]
    model_name = model_name_list[0]
    params = get_parameters(dataset_name, model_name,
                            dataset_dict[dataset_name], model_dict[model_name])

    classifierType = 'OR'
    myModel = params.network(Bottleneck,
                             [3, 4, 23, 3],
                             params.min_depth, params.max_depth,
                             params.crop_height, params.crop_width,
                             classifierType=classifierType,
                             num_classes=params.num_classes
                             )
    # fine tune
    resnet = models.resnet101(pretrained=True)
    myModel = fine_tune(resnet, myModel)
    del resnet
    # Loss Function
    if params.dataset_name == 'nyu':
        criterion = myModel.LossFunc(
            params.min_depth, params.max_depth, num_classes=params.num_classes,
            classifierType=classifierType, ignore_index=None)
    else:
        criterion = myModel.LossFunc(
            params.min_depth, params.max_depth, num_classes=params.num_classes,
            classifierType=classifierType, ignore_index=0)

    if mode == 'train':
        train(params, myModel, criterion)
    if mode == 'test':
        test(params, myModel, test_index)
    return

def train(params, myModel, criterion):
    torch.backends.cudnn.benchmark = True
    trainset = params.dataset_loader(params.home_image, params.home_depth, params.train_image_txt, params.train_depth_txt,
                                     params.original_height, params.original_width,
                                     params.resize_height, params.resize_width,
                                     params.crop_height, params.crop_width,
                                     params.min_depth, params.max_depth,
                                     'train'
                                     )
    trainloader = DataLoader(
        trainset, batch_size=params.batch_size, shuffle=True)

    validset = params.dataset_loader(params.home_image, params.home_depth, params.valid_image_txt, params.valid_depth_txt,
                                     params.original_height, params.original_width,
                                     params.resize_height, params.resize_width,
                                     params.crop_height, params.crop_width,
                                     params.min_depth, params.max_depth,
                                     'valid'
                                     )
    validloader = DataLoader(
        validset, batch_size=params.batch_size_valid, shuffle=False)
    # calculate total step
    num_training_samples = len(trainset)
    num_validation_samples = len(validset)
    steps_per_epoch = np.ceil(num_training_samples / params.batch_size).astype(np.int32)
    steps_per_epoch_valid = np.ceil(num_validation_samples / params.batch_size_valid).astype(np.int32)

    num_total_steps = params.num_epochs * steps_per_epoch
    num_total_steps_valid = params.num_epochs / params.interval_test * steps_per_epoch_valid
    print("training set: %d " % num_training_samples)
    print("validation set: %d" % num_validation_samples)
    print("train total steps: %d" % num_total_steps)
    print("valid total steps: %d" % num_total_steps_valid)
    print("Dataset {}".format(params.dataset_name))
    myModel.cuda()
    # print(repr(myModel))
    # filter manully
    base_modules = list(myModel.children())[:8]
    base_params = generate_all_params(base_modules)
    base_params = filter(lambda p: p.requires_grad, base_params)
    add_modules = list(myModel.children())[8:]
    add_weight_params = generate_weight_or_bias_params(add_modules, 'weight', 'all')
    add_bias_params = generate_weight_or_bias_params(add_modules, 'bias', 'all')
    # optimizer
    # optimizer = optim.Adam([
    #     {'params': base_params, 'lr': 0},
    #     {'params': add_params, 'lr': 0 * 10},
    # ], betas=(0.9, 0.95), weight_decay=0.0005)
    optimizer = optim.SGD([
        {'params': base_params, 'lr': 2e-4},
        {'params': add_weight_params, 'lr': 2e-4 * 10},
        {'params': add_bias_params, 'lr': 2e-4 * 20},
    ], momentum=0.9, weight_decay=0.0005)

    WRITER_DIR = params.log_path + '{}_{}/'.format(params.model_name, params.dataset_name)
    LOG_DIR = params.log_path + '{0}_{1}/{0}_{1}_loss.txt'.format(params.model_name, params.dataset_name)
    CHECKPOINT_DIR = params.checkpoint_path + '{}_{}/'.format(params.model_name, params.dataset_name)
    if os.path.isdir(CHECKPOINT_DIR) == 0:
        os.mkdir(CHECKPOINT_DIR)
    writer = SummaryWriter(WRITER_DIR)
    # GO!!!
    start_time = time.time()
    train_total_time = 0
    global_step = 0
    best_epoch = 0
    best_a1 = 0
    best_rmse = 0
    for epoch in range(params.num_epochs):
        torch.cuda.empty_cache()
        for step, (images, depths) in enumerate(trainloader):
            # adjust learning rate
            lr = adjust_learning_rate(optimizer, global_step + 1, num_total_steps, epoch + 1, params.num_epochs, params.learning_rate_mode)
            writer.add_scalar('lr', lr, global_step + 1)
            before_op_time = time.time()
            # input data
            images = Variable(images).cuda()
            depths = Variable(depths).cuda()
            # zero the grad
            optimizer.zero_grad()
            # forward
            # set the training mode, bn and dropout will be updated
            myModel.train()
            predict = myModel(images)
            # compute loss
            if params.model_name in ['contextattention']:
                image_loss, auxi_loss, total_loss = criterion(predict, depths, myModel.get_sim_map())
            else:
                image_loss, auxi_loss, total_loss = criterion(predict, depths)
            # backward
            total_loss.backward()
            # update parameters
            optimizer.step()
            # calculate time
            duration = time.time() - before_op_time
            fps = images.shape[0] / duration
            time_sofar = train_total_time / 3600
            time_left = (num_total_steps / (global_step + 1) - 1.0) * time_sofar
            # print loss
            print_interval = {'nyu': 100, 'kitti': 200}
            if (step + 1) % (steps_per_epoch // print_interval[params.dataset_name]) == 0:
                print_str = 'Epoch [{:>3}/{:>3}] | Step [{:>3}/{:>3}] | fps {:4.2f} | Image Loss: {:7.3f} | Auxi Loss: {:7.3f} | Total Loss: {:7.3f} | Time elapsed {:.2f}h | Time_left {:.2f}h'. \
                    format(epoch + 1, params.num_epochs, step + 1, steps_per_epoch, fps, image_loss, auxi_loss, total_loss, time_sofar, time_left)
                print(print_str)
                write_to_log(LOG_DIR, print_str)
            writer.add_scalar('total_loss', total_loss, global_step + 1)
            # if global_step == 1:
            #     writer.add_graph(myModel, (images, ))
            global_step += 1
            train_total_time += time.time() - before_op_time

        torch.cuda.empty_cache()

        if (epoch + 1) % params.interval_test == 0:
            print("<-------------Testing the model-------------->")
            test_total_time = 0
            with torch.no_grad():
                measure_list = ['a1', 'a2', 'a3', 'rmse', 'rmse_log', 'log10', 'abs_rel', 'sq_rel']
                measures = {key: 0 for key in measure_list}
                for index, (images, depths) in enumerate(validloader):
                    before_op_time = time.time()
                    # input data
                    images = Variable(images).cuda()
                    depths = Variable(depths).cuda()
                    # forward
                    # set the evaluation mode, bn and dropout will be fixed
                    myModel.eval()
                    predict = myModel(images)
                    pred_depth = myModel.inference(predict)
                    duration = time.time() - before_op_time
                    test_total_time += duration
                    # accuracy
                    new = compute_errors(depths, pred_depth)
                    for i in range(len(measure_list)):
                        measures[measure_list[i]] += new[measure_list[i]].item()
                    # display images
                    if index == 70:
                        display_figure(params, myModel, criterion, writer,
                                       images, depths, predict, epoch)

                measures_str = {key: '{:.5f}'.format(value / num_validation_samples) for key, value in measures.items()}
                fps = num_validation_samples / test_total_time
                print_str = 'The {}th epoch, fps {:4.2f} | {}'.format(epoch + 1, fps, measures_str)
                print(print_str)
                write_to_log(LOG_DIR, print_str)
                for i in range(len(measure_list)):
                    writer.add_scalar(measure_list[i], measures[measure_list[i]] / num_validation_samples, epoch + 1)
                # saving the model
                if epoch + 1 - params.interval_test != best_epoch:
                    last = CHECKPOINT_DIR + '{}.pkl'.format(epoch + 1 - params.interval_test)
                    if os.path.isfile(last):
                        os.remove(last)
                print("<+++++++++++++saving the model++++++++++++++>")
                now = CHECKPOINT_DIR + '{}.pkl'.format(epoch + 1)
                torch.save(myModel.state_dict(), now)

                now_a1 = measures[measure_list[0]]
                now_rmse = measures[measure_list[3]]
                if now_a1 >= best_a1 and now_rmse <= best_rmse:
                    last_best = CHECKPOINT_DIR + '{}.pkl'.format(best_epoch)
                    if os.path.isfile(last_best):
                        os.remove(last_best)
                    best_epoch = epoch + 1
                    best_a1 = now_a1
                    best_rmse = now_rmse
                elif now_a1 < best_a1 and now_rmse < best_rmse:
                    pass
                else:
                    best_epoch = epoch + 1
                    best_a1 = now_a1
                    best_rmse = now_rmse
    print("Finished training!")
    end_time = time.time()
    print("Spend time: %.2fh" % ((end_time - start_time) / 3600))
    return

def imshow(img, height, width, mode='image'):
    _, _, h, w = img.shape
    img = torchvision.utils.make_grid(img, nrow=width)
    if img.is_cuda:
        npimg = img.cpu().numpy()
    else:
        npimg = img.numpy()
    fig = plt.figure(figsize=(w // 80 * width, h // 50 * height))
    if mode == 'image':
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
    elif mode == 'depth':
        plt.imshow(npimg[0], cmap='plasma')
    else:
        plt.imshow(npimg[0], cmap='hot')
    return fig

def display_figure(params, myModel, criterion, writer, images, depths, predict, epoch):
    color = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
    marker = ['o', 'v', '^', '<', '>']
    # display image figure
    b, h, w = depths.shape[0], params.crop_height, params.crop_width
    images = F.interpolate(images, size=(h // 2, w // 2),
                           mode='bilinear', align_corners=True)
    fig1 = imshow(images, height=1, width=b, mode='image')
    del images
    writer.add_figure('figure1-images', fig1, epoch + 1)
    # display gt and pred figure
    depths = F.interpolate(depths, size=(h // 2, w // 2),
                           mode='nearest')
    if params.model_name in ['multiscale', 'attention']:
        output = []
        for i in range(len(predict)):
            scale = F.interpolate(myModel.inference(predict[i]), size=(
                h // 2, w // 2), mode='nearest', align_corners=True)
            output = output + [scale]
        output = torch.cat(output, 0)
    else:
        output = F.interpolate(myModel.inference(predict), size=(
            h // 2, w // 2), mode='nearest')
    fuse = torch.cat((depths, output), 0)
    fig2 = imshow(fuse, height=2, width=b, mode='depth')
    del fuse, output
    writer.add_figure('figure2-depths', fig2, epoch + 1)
    # display similarity figure
    if params.model_name in ['contextattention']:
        sim_map = myModel.get_sim_map().cpu()
        N = sim_map.shape[1]
        points = [N // 4, N // 2, 3 * N // 4]
        sim_pixels = sim_map[:, points]
        sim_pixels = sim_pixels.reshape(
            (b, len(points), h // 8 // myModel.scale, w // 8 // myModel.scale))
        sim_pixels = sim_pixels.permute((1, 0, 2, 3))
        sim_pixels = sim_pixels.reshape(
            (-1, 1, h // 8 // myModel.scale, w // 8 // myModel.scale))
        sim_pixels = F.interpolate(sim_pixels, size=(
            h // 2, w // 2), mode='bilinear', align_corners=True)
        fig3 = imshow(sim_pixels, height=2, width=b, mode='sim_map')
        writer.add_figure('figure3-pixel_attentions', fig3, epoch + 1)
        del sim_map
    # display image-level context vector figure
    if params.model_name in ['contextattention']:
        image_context = myModel.get_image_context().cpu().numpy()
        channel = image_context.shape[1]
        legends = ['sample_{}'.format(i) for i in range(b)]
        fig4 = plt.figure()
        for i in range(b):
            plt.plot(np.arange(channel),
                     image_context[i], color=color[i], marker=marker[i])
        plt.legend(labels=legends, loc='upper right')
        plt.xlabel('channel')
        writer.add_figure('figure4-image_contexts', fig4, epoch + 1)
    return

# ------------------------------------------------------------------------------------

def test(params, myModel, test_index):
    torch.backends.cudnn.benchmark = True
    validset = params.dataset_loader(params.home_image, params.home_depth, params.valid_image_txt, params.valid_depth_txt,
                                     params.original_height, params.original_width,
                                     params.resize_height, params.resize_width,
                                     params.crop_height, params.crop_width,
                                     params.min_depth, params.max_depth,
                                     'valid'
                                     )
    validloader = DataLoader(validset, batch_size=1, shuffle=False)
    num_validation_samples = len(validset)
    steps_per_epoch_valid = np.ceil(
        num_validation_samples / params.batch_size_valid).astype(np.int32)
    print("validation set: %d" % num_validation_samples)
    # load parameters
    CHECKPOINT_DIR = params.checkpoint_path + '{}_{}/'.format(params.model_name, params.dataset_name)
    SAVE_DIR = params.save_path
    LOG_DIR = params.save_path + '{0}_{1}_measures.txt'.format(params.model_name, params.dataset_name)
    myModel.load_state_dict(torch.load(CHECKPOINT_DIR + '{}.pkl'.format(test_index)))
    myModel.cuda()
    print("<-------------Testing the model-------------->")
    test_total_time = 0
    with torch.no_grad():
        colormap = {'nyu': 'jet', 'kitti': 'plasma', 'nust': 'plasma'}
        measure_list = ['a1', 'a2', 'a3', 'rmse', 'rmse_log', 'log10', 'abs_rel', 'sq_rel']
        measures = {key: 0 for key in measure_list}
        for index, (images, depths) in enumerate(validloader):
            before_op_time = time.time()
            # input data
            images = Variable(images).cuda()
            depths = Variable(depths).cuda()
            # forward
            # set the evaluation mode, bn and dropout will be fixed
            myModel.eval()
            predict = myModel(images)
            pred_depth = myModel.inference(predict)
            duration = time.time() - before_op_time
            test_total_time += duration
            # accuracy
            if params.dataset_name == 'kitti':
                new = compute_errors(depths[:, :, validset.crop[0]:validset.crop[1], validset.crop[2]:validset.crop[3]],
                                     pred_depth[:, :, validset.crop[0]:validset.crop[1], validset.crop[2]:validset.crop[3]])
            else:
                new = compute_errors(depths, pred_depth)

            image_context = myModel.get_image_context().cpu().numpy()
            print("saving the {}th depth".format(index))
            if os.path.isdir(SAVE_DIR) == 0:
                os.mkdir(SAVE_DIR)
            images = np.transpose(images.cpu().numpy().squeeze(), (1, 2, 0))
            depths = (depths.cpu() / params.max_depth).numpy().squeeze()
            output = (pred_depth.cpu() / params.max_depth).numpy().squeeze()
            plt.imsave(
                SAVE_DIR + '{:04}_rgb.png'.format(index), images)
            plt.imsave(SAVE_DIR + '{:04}_gt.png'.format(index),
                       depths, cmap=colormap[params.dataset_name])
            plt.imsave(SAVE_DIR + '{:04}_depth.png'.format(index),
                       output, cmap=colormap[params.dataset_name])
            for i in range(len(measure_list)):
                measures[measure_list[i]] += new[measure_list[i]].item()
        fps = num_validation_samples / test_total_time
        measures_str = {key: '{:.5f}'.format(value / num_validation_samples) for key, value in measures.items()}
        print_str = 'Testing the {} examples, fps {:4.2f} | {}'.format(
            num_validation_samples, fps, measures_str)
        print(print_str)
        write_to_log(LOG_DIR, print_str)
    return


if __name__ == '__main__':
    # sys.settrace(gpu_profile)
    main()
