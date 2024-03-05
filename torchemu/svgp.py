"""
Code: Implementation of Stochastic Variational GP scalable Gaussian Process regression.
Reference: https://docs.gpytorch.ai/en/latest/examples/04_Variational_and_Approximate_GPs/SVGP_Regression_CUDA.html
Author: Arrykrishna Mootoovaloo
Date: August 2023
Email: arrykrish@gmail.com
"""
import tqdm
import torch
from torch.utils.data import TensorDataset, DataLoader

# from gpytorch library
import gpytorch
from gpytorch.models import ApproximateGP
from gpytorch.variational import CholeskyVariationalDistribution, VariationalStrategy
from gpytorch.distributions import MultivariateNormal
from gpytorch.mlls import VariationalELBO
from gpytorch.likelihoods import GaussianLikelihood


def generate_dataloader(
    xinputs: torch.Tensor,
    targets: torch.Tensor,
    batch_size: int = 256,
    shuffle: bool = False,
):
    """
    Generates a dataloader given the inputs and the targets.

    Args:
        xinputs (torch.Tensor): the inputs to the model.
        targets (torch.Tensor): the target values of the regression model.
        batch_size (int, optional): the batch size to be used in the modelling. Defaults to 256.
        shuffle (bool, optional): option to shuffle the data. Defaults to False.
    """
    if torch.cuda.is_available():
        xinputs = xinputs.cuda()
        targets = targets.cuda()
    dataset = TensorDataset(xinputs, targets)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader


class GPModel(ApproximateGP):
    """
    The approximate Gaussian Process model.

    Args:
        inducing_points (torch.Tensor): the inducing points for the Gaussian Process model.
    """

    def __init__(self, inducing_points: torch.Tensor):
        variational_distribution = CholeskyVariationalDistribution(
            inducing_points.size(0)
        )
        variational_strategy = VariationalStrategy(
            self,
            inducing_points,
            variational_distribution,
            learn_inducing_locations=True,
        )
        super(GPModel, self).__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, xpoints: torch.Tensor) -> MultivariateNormal:
        """
        Pass the inputs to the Gaussian Process model

        Args:
            xpoints (torch.Tensor): the inputs to the model

        Returns:
            MultivariateNormal: a multivariate normal distribution
        """
        mean_x = self.mean_module(xpoints)
        covar_x = self.covar_module(xpoints)
        return MultivariateNormal(mean_x, covar_x)


class EmulatorSVGP:
    """
    An emulator using Stochastic Variational Gaussian Process.

    Args:
        train_dataloader (DataLoader): the train data loader.
        inducing (torch.Tensor): the inducing points.
    """

    def __init__(self, train_dataloader: DataLoader, inducing: torch.Tensor):
        self.train_dataloader = train_dataloader
        self.inducing = inducing
        self._postinit()

    def _postinit(self):
        self.ntrain = len(self.train_dataloader.dataset)
        self.nind = len(self.inducing)

        self.model = GPModel(inducing_points=self.inducing)
        self.likelihood = GaussianLikelihood()

        if torch.cuda.is_available():
            self.model = self.model.cuda()
            self.likelihood = self.likelihood.cuda()

    def training(self, lrate: float = 0.01, nepochs: int = 10):
        """
        Train the Gaussian Process model.

        Args:
            lrate (float, optional): The learning rate to use in the training. Defaults to 0.01.
            nepochs (int, optional): The number of epochs for training. Defaults to 10.
        """
        self.set_to_eval(False)

        optimizer = torch.optim.Adam(
            [
                {"params": self.model.parameters()},
                {"params": self.likelihood.parameters()},
            ],
            lr=lrate,
        )

        # Our loss object. We're using the VariationalELBO
        mll = VariationalELBO(self.likelihood, self.model, num_data=self.ntrain)

        epochs_iter = tqdm.notebook.tqdm(range(nepochs), desc="Epoch")
        for i in epochs_iter:
            # Within each iteration, we will go over each minibatch of data
            minibatch_iter = tqdm.notebook.tqdm(
                self.train_dataloader, desc="Minibatch", leave=False
            )
            for x_batch, y_batch in minibatch_iter:
                optimizer.zero_grad()
                output = self.model(x_batch)
                loss = -mll(output, y_batch)
                minibatch_iter.set_postfix(loss=loss.item())
                loss.backward()
                optimizer.step()

    def set_to_eval(self, option: bool):
        """
        Set the model to either the evaluation or training model.

        Args:
            option (bool): if True, model will be set to evaluation mode, otherwise it will be set to train mode.
        """
        if option:
            self.model.eval()
            self.likelihood.eval()
        else:
            self.model.train()
            self.likelihood.train()

    def calculate_accuracy(self, test_dataloader: DataLoader) -> torch.Tensor:
        """
        Calculates the accuracy of the predictions from the GP model.

        Args:
            test_dataloader (DataLoader): the test dataloader.

        Returns:
            torch.Tensor: the accuracy of the predictions.
        """
        self.set_to_eval(True)
        means = torch.tensor([0.0])
        with torch.no_grad():
            for x_batch, y_batch in test_dataloader:
                preds = self.model(x_batch)
                means = torch.cat([means, preds.mean.cpu()])
        means = means[1:]
        test_y = test_dataloader.dataset.tensors[1].cpu()
        accuracy = (means - test_y) / test_y
        print(
            "Test Mean Abosolute Error (MAE): {}".format(
                torch.mean(torch.abs(means - test_y.cpu()))
            )
        )
        return accuracy

    def calculate_prediction(
        self, testpoint: torch.Tensor, variance: bool = False
    ) -> torch.Tensor:
        """
        Calculate the prediction at a given test point in parameter space.

        Args:
            testpoint (torch.Tensor): the test point where we want to compute the prediction.
            variance (bool): option to output the variance. Defaults to False.

        Returns:
            torch.Tensor: the prediction.
        """
        self.set_to_eval(True)
        if torch.cuda.is_available():
            testpoint = testpoint.cuda()
        mean = self.model(testpoint.view(1, -1)).mean.data.cpu()
        if variance:
            var = self.model(testpoint.view(1, -1)).variance.data.cpu()
            return mean, var
        return mean
