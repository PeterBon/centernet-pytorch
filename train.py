# -------------------------------------#
#       对数据集进行训练
# -------------------------------------#
import os
import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm

from nets.centernet import CenterNet_Resnet50
from nets.centernet_training import focal_loss, reg_l1_loss
from utils.dataloader import CenternetDataset, centernet_dataset_collate
import yaml
from tensorboardX import SummaryWriter
import utils.message


# ---------------------------------------------------#
#   获得类
# ---------------------------------------------------#
def get_classes(classes_path):
    '''loads the classes'''
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def fit_one_epoch(net, epoch, epoch_size, epoch_size_val, gen, genval, Epoch, cuda):
    total_r_loss = 0
    total_c_loss = 0
    total_loss = 0
    val_loss = 0


    net.train()
    with tqdm(total=epoch_size, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3) as pbar:
        for iteration, batch in enumerate(gen):
            start_time = time.time()
            if iteration >= epoch_size:
                break
            with torch.no_grad():
                if cuda:
                    batch = [Variable(torch.from_numpy(ann).type(torch.FloatTensor)).cuda() for ann in batch]
                else:
                    batch = [Variable(torch.from_numpy(ann).type(torch.FloatTensor)) for ann in batch]

            batch_images, batch_hms, batch_whs, batch_regs, batch_reg_masks = batch

            optimizer.zero_grad()

            hm, wh, offset = net(batch_images)
            c_loss = focal_loss(hm, batch_hms)
            wh_loss = 0.1 * reg_l1_loss(wh, batch_whs, batch_reg_masks)
            off_loss = reg_l1_loss(offset, batch_regs, batch_reg_masks)

            loss = c_loss + wh_loss + off_loss

            total_loss += loss.item()
            total_c_loss += c_loss.item()
            total_r_loss += wh_loss.item() + off_loss.item()

            loss.backward()
            optimizer.step()

            waste_time = time.time() - start_time
            pbar.set_postfix(**{'total_r_loss': total_r_loss / (iteration + 1),
                                'total_c_loss': total_c_loss / (iteration + 1),
                                'lr': get_lr(optimizer),
                                's/step': waste_time})
            pbar.update(1)


    net.eval()
    print('Start Validation')
    with tqdm(total=epoch_size_val, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3) as pbar:
        for iteration, batch in enumerate(genval):
            if iteration >= epoch_size_val:
                break
            with torch.no_grad():
                if cuda:
                    batch = [Variable(torch.from_numpy(ann).type(torch.FloatTensor)).cuda() for ann in batch]
                else:
                    batch = [Variable(torch.from_numpy(ann).type(torch.FloatTensor)) for ann in batch]

                batch_images, batch_hms, batch_whs, batch_regs, batch_reg_masks = batch


                hm, wh, offset = net(batch_images)
                c_loss = focal_loss(hm, batch_hms)
                wh_loss = 0.1 * reg_l1_loss(wh, batch_whs, batch_reg_masks)
                off_loss = reg_l1_loss(offset, batch_regs, batch_reg_masks)
                loss = c_loss + wh_loss + off_loss
                val_loss += loss.item()


            pbar.set_postfix(**{'total_loss': val_loss / (iteration + 1)})
            pbar.update(1)


    # tensorboardX
    writer.add_scalars('loss', {'train': total_loss / (epoch_size + 1), 'val': val_loss / (epoch_size_val + 1)},
                       epoch)
    writer.add_scalar('lr', get_lr(optimizer), epoch)
    writer.flush()

    print('Finish Validation')
    print('Epoch:' + str(epoch + 1) + '/' + str(Epoch))
    print('Total Loss: %.4f || Val Loss: %.4f ' % (total_loss / (epoch_size + 1), val_loss / (epoch_size_val + 1)))

    print('Saving state, iter:', str(epoch + 1))
    # log.yaml
    avg_train_loss = total_loss / (epoch_size + 1)
    avg_train_loss = avg_train_loss.item()
    avg_val_loss = val_loss / (epoch_size_val + 1)
    avg_val_loss = avg_val_loss.item()
    log['epoch_number'] += 1
    log['Epoch%03d' % (epoch + 1)] = [avg_train_loss, avg_val_loss]
    if log['best_val_loss'] < 0 or avg_val_loss < log['best_val_loss']:
        log['best_val_loss'] = avg_val_loss
        torch.save(model.state_dict(), 'logs/best.pth')
    with open('logs/log.yaml', 'w', encoding='utf-8') as f:
        yaml.dump(log, f)

    torch.save(model.state_dict(), 'logs/last.pth')

    return val_loss / (epoch_size_val + 1)


if __name__ == "__main__":
    # hyp
    with open('model_data/hyp.yaml', encoding='utf-8') as f:
        hyp = yaml.load(f, Loader=yaml.FullLoader)

    # log
    log_dir = hyp.get('log_dir')
    if os.path.exists(os.path.join(log_dir, 'log.yaml')):
        with open(os.path.join(log_dir, 'log.yaml'), encoding='utf-8') as f:
            log = yaml.load(f, Loader=yaml.FullLoader)
    else:
        log = {'epoch_number': 0, 'best_val_loss': -1}
    # -------------------------------------------#
    #   输入图片的大小
    # -------------------------------------------#
    input_shape = hyp.get('input_shape')
    # -----------------------------#
    #   训练前一定要注意注意修改
    #   classes_path对应的txt的内容
    #   修改成自己需要分的类
    # -----------------------------#
    classes_path = hyp.get('classes_path')
    # ----------------------------------------------------#
    #   获取classes和数量
    # ----------------------------------------------------#
    class_names = get_classes(classes_path)
    num_classes = len(class_names)




    # ----------------------------------------------------#
    #   获取centernet模型
    # ----------------------------------------------------#
    model = CenterNet_Resnet50(num_classes)


    # ------------------------------------------------------#
    #   权值文件请看README，百度网盘下载
    # ------------------------------------------------------#
    model_path = hyp.get('model_path')
    if model_path:
        print('Loading weights into state dict...')
        model_dict = model.state_dict()
        pretrained_dict = torch.load(model_path)
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if np.shape(model_dict[k]) == np.shape(v)}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        print('Finished!')

    net = model.train()

    # -------------------------------#
    #   是否使用Cuda
    #   没有GPU可以设置成False
    # -------------------------------#
    Cuda = torch.cuda.is_available()
    if Cuda:
        net = torch.nn.DataParallel(model)
        cudnn.benchmark = True
        net = net.cuda()

    # ----------------------------------------------------#
    #   获得图片路径和标签
    # ----------------------------------------------------#
    annotation_path = hyp.get('annotation_path')
    # ----------------------------------------------------------------------#
    #   验证集的划分在train.py代码里面进行
    #   2007_test.txt和2007_val.txt里面没有内容是正常的。训练不会使用到。
    #   当前划分方式下，验证集和训练集的比例为1:9
    # ----------------------------------------------------------------------#
    val_split = 0.1
    with open(annotation_path) as f:
        lines = f.readlines()
    np.random.seed(10101)
    np.random.shuffle(lines)
    np.random.seed(None)
    num_val = int(len(lines) * val_split)
    num_train = len(lines) - num_val

    # tensorboardX
    writer = SummaryWriter(logdir='logs')
    if Cuda:
        graph_inputs = torch.from_numpy(np.random.rand(1, 3, input_shape[0], input_shape[1])).type(
            torch.FloatTensor).cuda()
    else:
        graph_inputs = torch.from_numpy(np.random.rand(1, 3, input_shape[0], input_shape[1])).type(torch.FloatTensor)
    writer.add_graph(model, (graph_inputs,))

    # ------------------------------------------------------#
    #   主干特征提取网络特征通用，冻结训练可以加快训练速度
    #   也可以在训练初期防止权值被破坏。
    #   Init_Epoch为起始世代
    #   Freeze_Epoch为冻结训练的世代
    #   Epoch总训练世代
    #   提示OOM或者显存不足请调小Batch_size
    # ------------------------------------------------------#

    Batch_size = hyp.get('batch_size')
    start_epoch = hyp.get('start_epoch')
    end_epoch = hyp.get('end_epoch')
    optimizer = optim.Adam([{'params': net.parameters(), 'initial_lr': hyp.get('lr')}], lr=hyp.get('lr'),
                           weight_decay=hyp.get('weight_decay'))
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=2, verbose=True)

    train_dataset = CenternetDataset(lines[:num_train], input_shape, num_classes, True)
    val_dataset = CenternetDataset(lines[num_train:], input_shape, num_classes, False)
    gen = DataLoader(train_dataset, batch_size=Batch_size, num_workers=12, pin_memory=True,
                     drop_last=True, collate_fn=centernet_dataset_collate)
    gen_val = DataLoader(val_dataset, batch_size=Batch_size, num_workers=12, pin_memory=True,
                         drop_last=True, collate_fn=centernet_dataset_collate)
    epoch_size = num_train // Batch_size
    epoch_size_val = num_val // Batch_size

    if hyp.get('freeze'):
        model.freeze_backbone()
    else:
        model.unfreeze_backbone()

    for epoch in range(start_epoch, end_epoch):
        val_loss = fit_one_epoch(net, epoch, epoch_size, epoch_size_val, gen, gen_val, end_epoch, Cuda)
        lr_scheduler.step(val_loss)

    writer.close()
    utils.message.send_email('训练完毕','训练完毕')
