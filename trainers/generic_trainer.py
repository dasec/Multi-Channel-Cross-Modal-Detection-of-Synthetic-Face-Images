# Class based on the implementation from Bob: https://www.idiap.ch/software/bob/ 
import copy
import os
import time
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

from torch.utils.tensorboard import SummaryWriter


class GenericTrainer(object):
    """
    Class to train a generic NN; all the parameters are provided in configs

    Attributes
    ----------
    network: :py:class:`torch.nn.Module`
            The network to train
    optimizer: :py:class:`torch.optim.Optimizer`
            Optimizer object to be used. Initialized in the config file. 

    device: str
            Device which will be used for training the model
    verbosity_level: int
            The level of verbosity output to stdout

    """

    def __init__(
        self,
        network,
        optimizer,
        compute_loss,
        learning_rate=0.0001,
        device="cpu",
        verbosity_level=2,
        tf_logdir="tf_logs",
        do_crossvalidation=False,
        save_interval=5,
    ):
        """ Init function . The layers to be adapted in the network is selected and the gradients are set to `True` 
        for the  layers which needs to be adapted. 

        Parameters
        ----------
        network: :py:class:`torch.nn.Module`
                The network to train
        device: str
                Device which will be used for training the model
        verbosity_level: int
                The level of verbosity output to stdout
        do_crossvalidation: bool
                If set to `True`, performs validation in each epoch and stores the best model based on validation loss.
        """
        self.network = network
        self.optimizer = optimizer
        self.compute_loss = compute_loss
        self.device = device
        self.learning_rate = learning_rate
        self.save_interval = save_interval

        self.do_crossvalidation = do_crossvalidation

        if self.do_crossvalidation:
            phases = ["train", "val"]
        else:
            phases = ["train"]
        self.phases = phases

        # Move the network to device
        self.network.to(self.device)

        self.tf_logger = SummaryWriter(log_dir=tf_logdir)

        # Setting the gradients to true for the layers which needs to be adapted

    def load_model(self, model_filename):
        """Loads an existing model

        Parameters
        ----------
        model_file: str
                The filename of the model to load

        Returns
        -------
        start_epoch: int
                The epoch to start with
        start_iteration: int
                The iteration to start with
        losses: list(float)
                The list of losses from previous training 

        """

        cp = torch.load(model_filename)
        self.network.load_state_dict(cp["state_dict"])
        start_epoch = cp["epoch"]
        start_iter = cp["iteration"]
        losses = cp["loss"]
        return start_epoch, start_iter, losses

    def save_model(self, output_dir, epoch=0, iteration=0, losses=None):
        """Save the trained network

        Parameters
        ----------
        output_dir: str
                The directory to write the models to
        epoch: int
                the current epoch
        iteration: int
                the current (last) iteration
        losses: list(float)
                The list of losses since the beginning of training 

        """

        saved_filename = "model_{}_{}.pth".format(epoch, iteration)
        saved_path = os.path.join(output_dir, saved_filename)
        print("Saving model to {}".format(saved_path))
        cp = {
            "epoch": epoch,
            "iteration": iteration,
            "loss": losses,
            "state_dict": self.network.cpu().state_dict(),
        }
        torch.save(cp, saved_path)

        self.network.to(self.device)

    def train(self, dataloader, n_epochs=25, output_dir="out", model=None):
        """Performs the training.

        Parameters
        ----------
        dataloader: :py:class:`torch.utils.data.DataLoader`
                The dataloader for your data
        n_epochs: int
                The number of epochs you would like to train for
        learning_rate: float
                The learning rate for Adam optimizer.
        output_dir: str
                The directory where you would like to save models 
        model: str
                The path to a pretrained model file to start training from; this is the PAD model; not the LightCNN model

        """

        # if model exists, load it
        if model is not None:
            start_epoch, start_iter, losses = self.load_model(model)
            print(
                "Starting training at epoch {}, iteration {} - last loss value is {}".format(
                    start_epoch, start_iter, losses[-1]
                )
            )
        else:
            start_epoch = 0
            start_iter = 0
            losses = []
            print("Starting training from scratch")

        for name, param in self.network.named_parameters():

            if param.requires_grad == True:
                print("Layer to be adapted from grad check : {}".format(name))

        # setup optimizer

        self.network.train(True)

        best_model_wts = copy.deepcopy(self.network.state_dict())

        best_loss = float("inf")

        # let's go
        for epoch in range(start_epoch, n_epochs):

            # in the epoch

            train_loss_history = []

            val_loss_history = []

            for phase in self.phases:

                if phase == "train":
                    self.network.train()  # Set model to training mode
                else:
                    self.network.eval()  # Set model to evaluate mode

                for i, data in enumerate(dataloader[phase], 0):

                    if i >= start_iter:

                        start = time.time()

                        # get data from dataset

                        img, labels = data

                        self.optimizer.zero_grad()

                        with torch.set_grad_enabled(phase == "train"):

                            loss = self.compute_loss(
                                self.network, img, labels, self.device
                            )

                            if phase == "train":

                                loss.backward()

                                self.optimizer.step()

                                train_loss_history.append(loss.item())
                            else:

                                val_loss_history.append(loss.item())

                        end = time.time()

                        print(
                            "[{}/{}][{}/{}] => Loss = {} (time spent: {}), Phase {}".format(
                                epoch,
                                n_epochs,
                                i,
                                len(dataloader[phase]),
                                loss.item(),
                                (end - start),
                                phase,
                            )
                        )

                        losses.append(loss.item())

            epoch_train_loss = np.mean(train_loss_history)

            print("Train Loss : {}  epoch : {}".format(epoch_train_loss, epoch))

            if self.do_crossvalidation:

                epoch_val_loss = np.mean(val_loss_history)

                print("Val Loss : {}  epoch : {}".format(epoch_val_loss, epoch))

                if phase == "val" and epoch_val_loss < best_loss:

                    best_loss = epoch_val_loss

                    best_model_wts = copy.deepcopy(self.network.state_dict())

            ########################################  <Logging> ###################################
            if self.do_crossvalidation:

                info = {"train_loss": epoch_train_loss, "val_loss": epoch_val_loss}
            else:

                info = {"train_loss": epoch_train_loss}

            # scalar logs

            for tag, value in info.items():
                self.tf_logger.add_scalar(
                    tag=tag, scalar_value=value, global_step=epoch + 1
                )

            # Log values and gradients of the parameters (histogram summary)

            for tag, value in self.network.named_parameters():
                tag = tag.replace(".", "/")
                try:
                    self.tf_logger.add_histogram(
                        tag=tag, values=value.data.cpu().numpy(), global_step=epoch + 1
                    )
                    self.tf_logger.add_histogram(
                        tag=tag + "/grad",
                        values=value.grad.data.cpu().numpy(),
                        global_step=epoch + 1,
                    )
                except:
                    pass

            ########################################  </Logging>  ###################################

            # do stuff - like saving models
            print("EPOCH {} DONE".format(epoch + 1))

            # comment it out after debugging

            # save the last model, and the ones in the specified interval
            if (epoch + 1) == n_epochs or epoch % self.save_interval == 0:
                self.save_model(
                    output_dir, epoch=(epoch + 1), iteration=0, losses=losses
                )

        # load the best weights

        self.network.load_state_dict(best_model_wts)

        # best epoch is 0

        self.save_model(output_dir, epoch=0, iteration=0, losses=losses)
