import os

import torch
from tqdm import tqdm

from utils.utils import get_lr

# 定义一个世代的训练函数
def one_epoch(model_train, model, yolo_loss, loss_history,
              optimizer, epoch, epoch_step, epoch_val_step, gen, gen_val,
              Epoch, Cuda, fp16, scaler, save_period, save_dir, local_rand=0):
    """

    :param model_train: 表示模型所处状态，True:训练状态，False：验证/推理状态
    :param model: 模型本身
    :param yolo_loss: 损失模块
    :param loss_history: 损失记录模块
    :param eval_callback: 评估模块
    :param optimizer: 优化器模块
    :param epoch:   当前处于训练/验证第几世代
    :param epoch_step: 每个训练世代的步长
    :param epoch_val_step: 每个验证世代的步长
    :param gen: 训练集数据加载器
    :param gen_val: 验证集数据加载器
    :param Epoch: 总的训练世代，取值为UnFreeze_Epoch
    :param Cuda: 是否启用Cuda：bool
    :param fp16: 是否启动半精度计算：bool
    :param scaler: 同fp16状态相关的参数
    :param save_period: 每多少世代保存一次
    :param save_dir: 保存的目录
    :param local_rand:
    :return:
    """
    # 初始化损失值
    loss = 0
    val_loss = 0

    # ---------------------------#
    # 开始进行训练集训练，一般包括以下几步：
    # 1.模型切换至训练模式
    # 2.清理之前的梯度
    # 3.前向传播，得到模型的输出
    # 4.根据模型的输出和真实结果，进行损失计算
    # 5.反向传播，计算梯度
    # 6.使用优化器更新模型参数
    # ---------------------------#

    if local_rand == 0:
        print('开始第', epoch + 1, '世代训练')
        # 设置训练进度显示，利用tqdm进度条库建立一个pbar进度条对象
        # tqdm进度条在循环中通常与update()方法一起使用，以便在每次迭代时更新进度
        pbar = tqdm(total=epoch_step, desc=f'Epoch: {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3)
    # 模型切换至训练模式
    model_train.train()
    for iteration, batch in enumerate(gen):
        if iteration >= epoch_step:
            break
        # 从当前批次batch中提取图像（images）和目标（targets）
        images, targets = batch[0], batch[1]
        with torch.no_grad():
            if Cuda:
                images = images.cuda(local_rand)
                targets = [ann.cuda(local_rand) for ann in targets]
        # ----------------------#
        # 清零梯度
        # ----------------------#
        optimizer.zero_grad()
        # 如果非半精度计算的话，开始进行前向传播、计算损失、反向传播的操作
        if not fp16:
            # ------------------#
            # 前向传播
            # ------------------#
            outputs = model_train(images)

            loss_value_all = 0
            # ------------------#
            # 计算损失
            # ------------------#
            for l in range(len(outputs)):
                loss_item = yolo_loss(l, outputs[l], targets)
                loss_value_all += loss_item
            loss_value = loss_value_all

            # -----------------#
            # 反向传播
            # loss_value通常是一个标量（scalar）张量，它代表了模型在当前批次数据上的损失。
            # 调用backward()方法后，PyTorch会自动计算loss_value关于模型中所有可训练参数（即需要优化的参数）的梯度，
            # 并将这些梯度存储在对应参数的.grad属性中
            # optimizer.step()会根据存储在模型参数.grad属性中的梯度来更新这些参数。
            # 具体来说，它会根据优化器内部设定的学习率和其他超参数来调整每个参数的值，以减小损失函数。
            # 在optimizer.step()被调用之前，你需要先调用optimizer.zero_grad()来清除之前计算得到的梯度（如果有的话），因为PyTorch会累积梯度
            # -----------------#
            loss_value.backward()
            optimizer.step()
        else:
            from torch.cuda.amp import autocast
            with autocast():
                # 前向传播
                outputs = model_train(images)
                # 计算损失
                loss_value_all = 0
                for l in range(len(outputs)):
                    loss_item = yolo_loss(l, outputs[l], targets)
                    loss_value_all += loss_item
                loss_value = loss_value_all
            # 反向传播
            scaler.scale(loss_value).backward()
            scaler.step(optimizer)
            scaler.update()

        # 计算截止当前步长的损失总值
        loss += loss_value.item()

        if local_rand == 0:
            pbar.set_postfix(**{'loss': loss / (iteration + 1),
                                'lr': get_lr(optimizer)})
            pbar.update(1)

    if local_rand == 0:
        pbar.close()
        print('结束第', epoch + 1, '世代训练')
    # 开始进行验证集推理
        print('开始第', epoch + 1, '世代验证')
        pbar = tqdm(total=epoch_val_step, desc=f'Epoch: {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3)
    # 模型切换至验证模式
    model_train.eval()
    for iteration, batch in enumerate(gen_val):
        if iteration >= epoch_val_step:
            break

        images, targets = batch[0], batch[1]
        with torch.no_grad():
            if Cuda:
                images = images.cuda(local_rand)
                targets = [ann.cuda(local_rand) for ann in targets]
            # 清理梯度
            optimizer.zero_grad()
            # 前向传播
            outputs = model_train(images)
            # 计算损失
            loss_value_all = 0
            for l in range(len(outputs)):
                loss_item = yolo_loss(l, outputs[l], targets)
                loss_value_all += loss_item
            loss_value = loss_value_all
        val_loss += loss_value.item()

        if local_rand == 0:
            pbar.set_postfix(**{'val_loss': val_loss / (iteration + 1)})
            pbar.update(1)

    if local_rand == 0:
        pbar.close()
        print('结束第', epoch + 1, '世代验证')

        loss_history.append_loss(epoch + 1, loss / epoch_step, val_loss / epoch_val_step)
        # eval_callback.on_epoch_end(epoch + 1, model_train)
        print('Epoch:' + str(epoch + 1) + '/' + str(Epoch))
        print('Total Loss: %.3f || Val loss: %.3f' %(loss / epoch_step, val_loss / epoch_val_step))

        # ----------------------------------#
        # 保存权值
        # ----------------------------------#
        if (epoch + 1) % save_period == 0 or epoch + 1 ==Epoch:
            torch.save(model.state_dict(), os.path.join(save_dir, "ep%03d-loss%.3f-val_loss%.3f.pth" % (epoch + 1, loss / epoch_step, val_loss / epoch_val_step)))

        if len(loss_history.val_losses) <= 1 or (val_loss / epoch_val_step) <= min(loss_history.val_losses):
            print('Save best model to best_epoch_weights.pth')
            torch.save(model.state_dict(), os.path.join(save_dir, "best_epoch_weights.pth"))

        torch.save(model.state_dict(), os.path.join(save_dir, "last_epoch_weights.pth"))



