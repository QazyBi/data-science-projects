import argparse
import os
import shutil
import math
import torch
# import torch.nn as nn
import torch.utils.data as dt

from carvana_dataset import CarvanaDataset
from model import SegmenterModel

from torch.autograd import Variable
import torch.optim as optim
from tensorboardX import SummaryWriter

from tqdm.auto import tqdm
import numpy as np


LOG_DIR = './log/'
TRAIN_DIR = './data/train/'
TRAIN_MASKS_DIR = './data/train_masks/'
TEST_DIR = './data/test/'
TEST_MASKS_DIR = './data/test_masks/'

if not os.path.exists(LOG_DIR):
    os.mkdir(LOG_DIR)
else:
    shutil.rmtree(LOG_DIR)

tb_writer = SummaryWriter(log_dir='log')

DEVICE_ID = 0
DEVICE = torch.device(f"cuda:{DEVICE_ID}")
torch.cuda.set_device(DEVICE_ID)

def cyclical_lr(stepsize, min_lr=3e-4, max_lr=3e-3):

    # Scaler: we can adapt this if we do not want the triangular CLR
    scaler = lambda x: 1.

    # Lambda function to calculate the LR
    def lr_lambda(it):
        return min_lr + (max_lr - min_lr) * relative(it, stepsize)

    # Additional function to see where on the cycle we are
    def relative(it, stepsize):
        cycle = math.floor(1 + it / (2 * stepsize))
        x = abs(it / stepsize - 2 * cycle + 1)
        return max(0, (1 - x)) * scaler(cycle)

    return lr_lambda


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=5)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    args = parser.parse_args()
    return args


def main():
    args = parse_arguments()

    model = SegmenterModel().to(DEVICE)  # Модель
    criterion = torch.nn.BCEWithLogitsLoss().to(DEVICE)  # Лосс
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)  # Алгоритм оптимизации

    ds = CarvanaDataset(TRAIN_DIR, TRAIN_MASKS_DIR)  # Обучающая выборка
    ds_test = CarvanaDataset(TEST_DIR, TEST_MASKS_DIR)  # Тестовая выборка

    # Инструменты для подгрузки тензоров с данными
    dl      = dt.DataLoader(ds, shuffle=True,
                            num_workers=2,
                            batch_size=args.batch_size)
    dl_test = dt.DataLoader(ds_test, shuffle=False,
                            num_workers=2,
                            batch_size=args.batch_size)

    step_size = 4 * len(dl)
    clr = cyclical_lr(step_size)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, [clr])

    ct = 0
    for child in model.children():
        if ct < 3:
            for param in child.parameters():
                param.requires_grad = False
        ct += 1

    global_iter = 0
    for epoch in range(0, args.n_epochs):
        print("Current epoch: ", epoch)
        epoch_loss = 0
        model.train(True)
        for i, (input_batch, target_batch) in enumerate(tqdm(dl)):
            optimizer.zero_grad()

            input_batch = Variable(input_batch).cuda()
            target_batch = Variable(target_batch).cuda()
            output_batch = model(input_batch)

            loss = criterion(output_batch, target_batch)
            loss.backward()

            optimizer.step()
            global_iter += 1
            epoch_loss += loss.item()
        epoch_loss = epoch_loss / float(len(ds))
        print("Epoch loss", epoch_loss)
        tb_writer.add_scalar('Loss/Train', epoch_loss, epoch)

        print("Make test")
        test_loss = 0
        model.train(False)
        tb_out = np.random.choice(range(0, len(dl_test)), 3)
        for i, (input_batch, target_batch) in enumerate(tqdm(dl_test)):
            input_batch = input_batch.to(DEVICE)
            target_batch = target_batch.to(DEVICE)
            with torch.no_grad():
                output_batch = model(input_batch)
            loss = criterion(output_batch, target_batch)
            test_loss += loss.item()

            for img_id, checkpoint in enumerate(tb_out):
                if checkpoint == i:
                    tb_writer.add_image(f'Image/Test_input_{img_id}',
                                        input_batch[0].cpu(),
                                        epoch)
                    tb_writer.add_image(f'Image/Test_target_{img_id}',
                                        target_batch[0].cpu(),
                                        epoch)
                    tb_writer.add_image(f'Image/Test_output_{img_id}',
                                        output_batch[0].cpu() > 0,
                                        epoch)

        test_loss = test_loss / float(len(ds_test))
        print("Test loss", test_loss)
        tb_writer.add_scalar('Loss/Test', test_loss, epoch)
        torch.save(model.state_dict(), "unet_dump_recent")

        try:
            torch.save(model.state_dict(), "drive/MyDrive/hw6_sber/unet_dump_recent")
        except:
            pass

        scheduler.step()


if __name__ == '__main__':
    main()
