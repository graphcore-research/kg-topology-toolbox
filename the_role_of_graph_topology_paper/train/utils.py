# Copyright (c) 2023 Graphcore Ltd. All rights reserved.

import logging
import os
import pickle

from torch.optim import lr_scheduler


def load_or_none(file):
    if os.path.isfile(file):
        return pickle.load(open(file, "rb"))
    return None


def create_logger():
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(asctime)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger = logging.getLogger()
    return logger


def get_lr_scheduler(optimizer, scheduler_type, gamma=0.9, num_epochs=None):
    if not scheduler_type:
        return None
    if scheduler_type == "linear":
        scheduler = lr_scheduler.LinearLR(
            optimizer,
            start_factor=1.0,
            end_factor=0.0,
            total_iters=num_epochs,
            last_epoch=-1,
        )
    elif scheduler_type == "exponential":
        scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=gamma, last_epoch=-1)
    else:
        raise ValueError("Unknown scheduler_type:", scheduler_type)

    # Initialize step as PopTorch does not call optimizer.step() explicitly
    optimizer._step_count = 1

    return scheduler
