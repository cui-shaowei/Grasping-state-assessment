import torch
import torch.nn as nn
import os
import shutil
import time
import torch.nn.parallel
from options import Options
# from gelsight_feature_loader import MyFeatureDataset
from xela_dataloader import MyDataset
from utils import Bar, Logger, AverageMeter, savefig, ACC
from models.models import *
import cv2
from network_utils import init_weights_xavier
from sklearn.metrics import accuracy_score
import torch.nn.functional as F
from torch.utils import data
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import matplotlib.pyplot as plt

def main():
    opt = Options().parse()
    start_epoch = opt.start_epoch  # start from epoch 0 or last checkpoint epoch
    opt.phase = 'train'
    transform_v = transforms.Compose([transforms.Resize([opt.cropWidth, opt.cropHeight]),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    transform_t = transforms.Compose([transforms.Resize([4, 4]),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    trainset = MyDataset('xeladataset_train2.csv',5,10,transform_v,transform_t)
    train_loader = torch.utils.data.DataLoader(
        dataset=trainset,
        batch_size=opt.batchSize,
        shuffle=True,
        num_workers=int(opt.workers)
    )
    opt.phase = 'val'
    validset = MyDataset('xeladataset_test2.csv',5,10,transform_v,transform_t)
    val_loader = torch.utils.data.DataLoader(
        dataset=validset,
        batch_size=opt.batchSize,
        shuffle=False,
        num_workers=int(opt.workers)
    )

    # acc_list = []
    # for i in range(10):

    # Model
    if opt.model_arch == 'early_fusion':
        model = EarlyFusion(preTrain='resnet',fc_early_dim=64,LSTM_layers=2,LSTM_units=64,LSTM_dropout=0.5,num_classes=2,dropout_fc=0.2)
    elif opt.model_arch == 'early_fusionTA':
        model = EarlyFusionTA0(preTrain='resnet',fc_early_dim=64,T=8,LSTM_layers=2,LSTM_units=64,LSTM_dropout=0.5,num_classes=2,dropout_fc=0.2)
    elif opt.model_arch == 'Attearly_fusionTA':
        model=AttEarlyFusionTA(preTrain='resnet',fc_early_dim=64,T=8,LSTM_layers=2,LSTM_units=64,LSTM_dropout=0.5,num_classes=2,dropout_fc=0.2)
    elif opt.model_arch == 'Attearly_fusion':
        model= AttEarlyFusion(preTrain='resnet',fc_early_dim=64,T=8,LSTM_layers=2,LSTM_units=64,LSTM_dropout=0.5,num_classes=2,dropout_fc=0.2)
    elif opt.model_arch == 'LateFusion':
        model = LateFusion(preTrain='resnet', fc_early_dim=64, T=8, LSTM_layers=2, LSTM_units=64,
                                      LSTM_dropout=0.5, num_classes=2, dropout_fc=0.2)
    elif opt.model_arch == 'ModalFN':
        model = ModalFN(preTrain='resnet', fc_early_dim=64, T=8, LSTM_layers=2, LSTM_units=64,
                                      LSTM_dropout=0.5, num_classes=2, dropout_fc=0.2)
    elif opt.model_arch == 'ModalFNAtt0':
        model = ModalFNAtt0(preTrain='resnet', fc_early_dim=64, T=8, LSTM_layers=2, LSTM_units=64,
                                      LSTM_dropout=0.5, num_classes=2, dropout_fc=0.2)
    elif opt.model_arch == 'MARN':
        model = MARN(preTrain='resnet', fc_early_dim=64, T=8, cell_size=64, in_size=64,
                                      hybrid_in_size=64,num_atts=4, num_classes=2, dropout_fc=0.2)
    elif opt.model_arch == 'VTFSA_LSTM':
        model = VTFSA_LSTM(visual_cnn_out_dim=(7,7,512),tactile_cnn_out_dim=(7,7,512),lstm_hidden_layers = 2,lstm_hidden_nodes = 64,dropout_p_lstm=0.2,dropout_p_fc=0.5,encoder_fc_dim=64,fc_hidden_dim=64,num_classes=2)
    elif opt.model_arch == 'C3D':
        model = C3D(v_dim=3*5, img_xv=opt.cropWidth, img_yv=opt.cropHeight, drop_p_v=0.2, fc_hidden_v=256, ch1_v=16,ch2_v=24,ch1_t=8,ch2_t=12,t_dim=3*10, img_xt=4, img_yt=4, drop_p_t=0.2, fc_hidden_t=64,fc_hidden_1=128,num_classes=3)
    if opt.use_cuda:
        model = torch.nn.DataParallel(model).cuda()
    else:
        model.to( torch.device('cpu') )
    print('Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))

    # Loss and optimizer
    # criterion = nn.CrossEntropyLoss(reduction='sum')
    # optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr,betas=(0.9, 0.999), eps=1e-08,weight_decay=opt.weight_decay)
    optimizer = torch.optim.SGD(model.parameters(), lr=opt.lr)
    # model.apply(init_weights_xavier)
    # Resume
    title = opt.name
    if opt.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isfile(opt.resume), 'Error: no checkpoint directory found!'
        opt.checkpoint = os.path.dirname(opt.resume)
        checkpoint = torch.load(opt.resume)
        best_acc = checkpoint['best_acc']
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        logger = Logger(os.path.join(opt.checkpoint, 'log.txt'), title=title, resume=True)
    else:
        logger = Logger(os.path.join(opt.checkpoint, 'log.txt'), title=title)
        logger.set_names(['Learning Rate', 'Train Loss', 'Valid Loss', 'Valid PSNR.'])

    if opt.evaluate:
        print('\nEvaluation only')
        val_loss, val_psnr = valid(val_loader, model, start_epoch, opt.use_cuda)
        print(' Test Loss:  %.8f, Test PSNR:  %.2f' % (val_loss, val_psnr))
        return

    # Train and val

    best_acc = 0
    train_acc_list=[]
    train_loss_list=[]
    test_acc_list=[]
    test_loss_list=[]
    for epoch in range(start_epoch, opt.epochs):
        # adjust_learning_rate(optimizer, epoch, opt)
        # torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=50, verbose=False,
        #                                            threshold=0.0001, threshold_mode='rel', cooldown=20, min_lr=0,
        #                                            eps=1e-08)
        # if epoch < 5:
        #     opt.lr=0.0001
        # elif epoch >=5 and epoch <10:
        #     opt.lr=0.00001
        # elif epoch >= 10 and epoch <20:
        #     opt.lr=0.000001
        # elif epoch >= 20 and epoch < 50:
        #     opt.lr=0.0000001
        # else:
        #     opt.lr=0.00000001

        print('\nEpoch: [%d | %d] LR: %f' % (epoch + 1, opt.epochs, opt.lr))

        train_loss,train_acc = train(train_loader, model,  optimizer, epoch, opt.use_cuda)
        test_loss, test_acc= valid(val_loader, model,  epoch, opt.use_cuda)

        # append logger file
        logger.append([opt.lr, train_loss, test_loss, test_acc])
        train_acc_list.append(train_acc)
        train_loss_list.append(train_loss)
        test_loss_list.append(test_loss)
        test_acc_list.append(test_acc)
        # save model
        is_best = test_acc > best_acc
        best_acc = max(test_acc, best_acc)
        save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'acc': test_acc,
                'best_acc': best_acc,
                'optimizer' : optimizer.state_dict(),
            }, is_best, checkpoint=opt.checkpoint)
        print('Best acc:')
        print(best_acc)

        # acc_list.append(best_acc)
    # ave_acc=0
    # for i in range(len(acc_list)):
    #     ave_acc += acc_list[i]
    # print("average acc:",ave_acc/len(acc_list))
    logger.close()
    np.save(
        'XELA_results/train_acc_' + opt.model_arch + str(opt.batchSize) + '_' + str(opt.lr) + '.npy',
        train_acc_list)
    np.save('XELA_results/train_loss_' + opt.model_arch + str(opt.batchSize) + '_' + str(opt.lr) + '.npy', train_loss_list)
    np.save(
        'XELA_results/test_acc_' + opt.model_arch + str(opt.batchSize) + '_' + str(opt.lr) + '.npy',
        test_acc_list)
    np.save(
        'XELA_results/test_loss_' + opt.model_arch + str(opt.batchSize) + '_' + str(opt.lr) + '.npy',
        test_loss_list)
    plt.plot(train_loss_list)
    plt.plot(train_acc_list)
    plt.plot(test_loss_list)
    plt.plot(test_acc_list)

    plt.show()
    # logger.plot()
    # savefig(os.path.join(opt.checkpoint, 'log'+'.eps'))







def train(trainloader, model, optimizer, epoch, use_cuda):
    # switch to train mode
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    avg_acc = AverageMeter()
    psnr_input = AverageMeter()
    end = time.time()

    bar = Bar('Processing', max=len(trainloader))
    for batch_idx, (x_visual,x_tactile, targets) in enumerate(trainloader):
        # measure data loading time
        data_time.update(time.time() - end)

        x_tactile,x_visual, targets = torch.autograd.Variable(x_tactile),torch.autograd.Variable(x_visual), torch.autograd.Variable(targets)
        if use_cuda:
            # inputs = inputs.cuda()
            x_tactile=x_tactile.cuda()
            x_visual=x_visual.cuda()
            targets = targets.cuda(non_blocking=True)

        # compute output
        outputs = model(x_visual,x_tactile)
        loss = F.cross_entropy(outputs, targets, reduction='mean')
        # print(loss)
        # print(outputs)
        y_pred = torch.max(outputs, 1)[1]  # y_pred != output
        # print(y_pred)
        # print(targets)
        acc =  accuracy_score(y_pred.cpu().data.numpy(), targets.cpu().data.numpy())
        # psnr_i = PSNR(inputs, targets)



        # measure the result
        losses.update(loss.item(), x_tactile.size(0))
        avg_acc.update(acc, x_tactile.size(0))
        # psnr_input.update(psnr_i, inputs.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress | PSNR: {psnr: .4f} | PSNR(input): {psnr_in: .4f}
        bar.suffix  = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f}| ACC(input): {acc: .4f}'.format(
                    batch=batch_idx + 1,
                    size=len(trainloader),
                    data=data_time.avg,
                    bt=batch_time.avg,
                    total=bar.elapsed_td,
                    eta=bar.eta_td,
                    loss=losses.avg,
                    acc=avg_acc.avg,
                    # psnr_in=psnr_input.avg
                    )
        bar.next()
    bar.finish()
    return losses.avg,avg_acc.avg


def valid(testloader, model, epoch, use_cuda):
    # switch to train mode
    model.eval()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    avg_acc = AverageMeter()
    psnr_input = AverageMeter()
    end = time.time()

    bar = Bar('Processing', max=len(testloader))
    for batch_idx, (x_visual, x_tactile, targets) in enumerate(testloader):
        # measure data loading time
        data_time.update(time.time() - end)

        x_tactile, x_visual, targets = torch.autograd.Variable(x_tactile), torch.autograd.Variable(
            x_visual), torch.autograd.Variable(targets)
        if use_cuda:
            # inputs = inputs.cuda()
            x_tactile = x_tactile.cuda()
            x_visual = x_visual.cuda()
            targets = targets.cuda(non_blocking=True)

        # compute output
        outputs = model(x_visual,x_tactile)
        loss = F.cross_entropy(outputs, targets)
        y_pred = torch.max(outputs, 1)[1]  # y_pred != output
        acc =  accuracy_score(y_pred.cpu().data.numpy(), targets.cpu().data.numpy())
        # psnr_i = PSNR(inputs, targets)

        # measure the result
        losses.update(loss.item(), x_tactile.size(0))
        avg_acc.update(acc, x_tactile.size(0))
        # psnr_input.update(psnr_i, inputs.size(0))


        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress | PSNR: {psnr: .4f} | PSNR(input): {psnr_in: .4f}
        bar.suffix = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f}| ACC(input): {acc: .4f}'.format(
            batch=batch_idx + 1,
            size=len(testloader),
            data=data_time.avg,
            bt=batch_time.avg,
            total=bar.elapsed_td,
            eta=bar.eta_td,
            loss=losses.avg,
            acc=avg_acc.avg,
            # psnr_in=psnr_input.avg
        )
        bar.next()
    bar.finish()
    return (losses.avg, avg_acc.avg)


def adjust_learning_rate(optimizer, epoch, opt):
    if epoch % opt.schedule ==0 and epoch !=0 :
        opt.lr *= opt.gamma
        for param_group in optimizer.param_groups:
            param_group['lr'] = opt.lr


def save_checkpoint(state, is_best, checkpoint='checkpoint', filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'model_best.pth.tar'))


if __name__ == "__main__":
    main()
