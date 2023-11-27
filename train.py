#!/usr/bin/env python
# encoding: utf-8
import os, sys
import pkg_resources
import torch
import numpy
from options.train_options import TrainOptions

from trainers.generic_trainer import GenericTrainer
from trainers.detector_trainer_template import TrainerConfig

if __name__ == "__main__":

    configuration = TrainerConfig(TrainOptions().parse())

    batch_size = configuration.args.batch_size
    num_workers = configuration.args.num_workers
    epochs = configuration.args.epochs
    learning_rate = configuration.args.lr
    save_interval = configuration.args.save_interval
    seed = configuration.args.seed
    output_dir = configuration.args.output_dir
    gpu_id = int(configuration.args.gpu_id)
    use_gpu = gpu_id >= 0
    verbosity_level = 0
    do_crossvalidation = configuration.DO_CROSS_VALIDATION

    # load configuration file

    # use new interface
    if use_gpu and torch.cuda.is_available(): 
        device = torch.device("cuda:{0}".format(gpu_id))
    else: 
        device = torch.device("cpu")

    # process on the arguments / options
    torch.manual_seed(seed)
    if use_gpu:
        torch.cuda.manual_seed_all(seed)
    if torch.cuda.is_available() and not use_gpu:
        print("You have a CUDA device, so you should probably run with --use-gpu")

    print("Device used for training = {}".format(device))

    # Which device to use is figured out at this point, no need to use `use-gpu` flag anymore
    # get data
    if hasattr(configuration, "dataset"):

        dataloader = {}

        dataloader["train"] = torch.utils.data.DataLoader(
            configuration.dataset["train"],
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=True,
        )

        dataloader["val"] = torch.utils.data.DataLoader(
            configuration.dataset["val"],
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=True,
        )

        print(
            "There are {} training samples".format(
                len(configuration.dataset["train"])
            )
        )
        print(
            "There are {} validation samples".format(
                len(configuration.dataset["val"])
            )
        )

    else:
        print("Please provide a dataset in your configuration file !")
        sys.exit()

    assert hasattr(configuration, "optimizer")

    # train the network
    if hasattr(configuration, "network"):
        trainer = GenericTrainer(
            configuration.network,
            configuration.optimizer,
            configuration.compute_loss,
            learning_rate=learning_rate,
            device=device,
            verbosity_level=verbosity_level,
            tf_logdir=output_dir + "/tf_logs",
            do_crossvalidation=do_crossvalidation,
            save_interval=save_interval,
        )
        trainer.train(dataloader, n_epochs=epochs, output_dir=output_dir, model=None) # implement model to support loading from existing weights (non ImageNet weights)
    else:
        print("Please provide a network in your configuration file !")
        sys.exit()
