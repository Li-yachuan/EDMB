from utils import step_lr_scheduler, Averagvalue, cross_entropy_loss, norm_loss, save_checkpoint
import time
import torchvision
from os.path import join
import os
import torch
from torch.distributions import Normal, Independent


def train(train_loader, model, optimizer, epoch, args):
    optimizer = step_lr_scheduler(optimizer, epoch, args.stepsize)
    save_dir = join(args.savedir, 'epoch-%d-training-record' % epoch)
    os.makedirs(save_dir, exist_ok=True)

    batch_time = Averagvalue()
    data_time = Averagvalue()
    losses = Averagvalue()

    # switch to train mode
    model.train()
    print(epoch,
          "Pretrained lr:", optimizer.state_dict()['param_groups'][0]['lr'],
          "Unpretrained lr:", optimizer.state_dict()['param_groups'][2]['lr'])

    end = time.time()
    epoch_loss = []
    counter = 0
    for i, (image, label) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        image, label = image.cuda(), label.cuda()

        outputs = model(image)

        counter += 1
        if isinstance(outputs, list) and len(outputs) == 5:
            mean, std, local_mean, local_std, global_edge = outputs

            outputs_dist = Independent(Normal(loc=mean, scale=std + 0.001), 1)
            outputs = torch.sigmoid(outputs_dist.rsample())

            local_outputs_dist = Independent(Normal(loc=local_mean, scale=local_std + 0.001), 1)
            local_outputs = torch.sigmoid(local_outputs_dist.rsample())

            BCE_loss = cross_entropy_loss(outputs, label, args.loss_lmbda) \
                       + 0.4 * cross_entropy_loss(local_outputs, label, args.loss_lmbda)

            Norm_loss = norm_loss(mean, std) + 0.4 * norm_loss(local_mean, local_std)

            loss = BCE_loss + 0.01 * Norm_loss

            outputs = [outputs, local_outputs]

        elif isinstance(outputs, list) and len(outputs) == 3:
            # edge, local_edge, global_edge
            loss = cross_entropy_loss(outputs[0], label, args.loss_lmbda) + \
                   0.4 * cross_entropy_loss(outputs[1], label, args.loss_lmbda)

        else:
            loss = cross_entropy_loss(outputs, label, args.loss_lmbda)

        loss = loss / args.itersize
        loss.backward()

        if counter == args.itersize:
            optimizer.step()
            optimizer.zero_grad()
            counter = 0

        losses.update(loss, image.size(0))
        epoch_loss.append(loss)
        batch_time.update(time.time() - end)
        end = time.time()
        if i % args.print_freq == 0:
            info = 'Epoch: [{0}/{1}][{2}/{3}] '.format(epoch, args.maxepoch, i, len(train_loader)) + \
                   'Time {batch_time.val:.2f} (avg:{batch_time.avg:.2f}) '.format(batch_time=batch_time) + \
                   'Loss {loss.val:.2f} (avg:{loss.avg:.2f}) '.format(loss=losses)
            print(info)
            label[label == 2] = 0.5
            # if args.batch_size > 4:
            #     outputs = outputs[:4, ...]
            #     label = label[:4, ...]
            if isinstance(outputs, list):
                outputs.append(label)
                outputs = torch.cat(outputs, dim=0)
            else:
                outputs = torch.cat([outputs, label], dim=0)

            torchvision.utils.save_image(outputs, join(save_dir, "iter-%d.jpg" % i), nrow=args.batch_size)

    # save checkpoint
    save_checkpoint({
        'epoch': epoch,
        'state_dict': model.state_dict(),
    }, filename=join(save_dir, "epoch-%d-checkpoint.pth" % epoch))
