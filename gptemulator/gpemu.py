import torch
import gpytorch
import tqdm

POWER = 2


class PreWhiten(object):
    def __init__(self, xinputs: torch.tensor):
        # compute the covariance of the inputs (ndim x ndim)
        self.cov_train = torch.cov(xinputs.t())
        self.ndim = xinputs.shape[1]

        # compute the Cholesky decomposition of the matrix
        self.chol_train = torch.linalg.cholesky(self.cov_train)

        # compute the mean of the sample
        self.mean_train = torch.mean(xinputs, axis=0).view(1, self.ndim)

    def x_transformation(self, point: torch.tensor) -> torch.tensor:
        """Pre-whiten the input parameters.

        Args:
            point (torch.tensor): the input parameters.

        Returns:
            torch.tensor: the pre-whitened parameters.
        """

        # ensure the point has the right dimensions
        point = point.view(-1, self.ndim)

        # calculate the transformed training points
        transformed = torch.linalg.inv(self.chol_train) @ (point - self.mean_train).t()

        return transformed.t()


class ExactGPModelAll(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModelAll, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.mean_module.initialize(constant=0.0)
        self.mean_module.constant.requires_grad = False
        ndim = train_x.shape[1]
        kernel_1 = gpytorch.kernels.RBFKernel(
            ard_num_dims=ndim, active_dims=range(ndim)
        )
        kernel_2 = gpytorch.kernels.PolynomialKernel(
            num_dimensions=ndim, power=POWER, active_dims=range(ndim)
        )
        self.covar_module = gpytorch.kernels.ScaleKernel(kernel_1 + kernel_2)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class GPModel(PreWhiten):

    def __init__(self, inputs, yTransform):

        self.inputs = torch.from_numpy(inputs)
        self.yTransform = yTransform
        PreWhiten.__init__(self, self.inputs)

        self.train_x = PreWhiten.x_transformation(self, self.inputs)
        self.train_x = self.train_x.to(torch.float32)
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood()
        self.model = ExactGPModelAll(
            self.train_x, self.yTransform.train_y, self.likelihood
        )

    def training(self, ntrain, lr=0.1, noise=1e-5, verbose=True):

        # Find optimal model hyperparameters
        self.model.train()
        self.likelihood.train()

        # Use the adam optimizer
        if noise is not None:
            self.model.likelihood.noise = noise
            training_parameters = [
                p
                for name, p in self.model.named_parameters()
                if not name.startswith("likelihood")
            ]
        else:
            training_parameters = self.model.parameters()

        # Use the adam optimizer
        optimizer = torch.optim.Adamax(training_parameters, lr=lr)

        # "Loss" for GPs - the marginal log likelihood
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model)

        record_loss = []
        with tqdm.trange(ntrain, disable=not verbose) as bar:
            for i in bar:
                optimizer.zero_grad()
                out = self.model(self.train_x)
                loss = -mll(out, self.yTransform.train_y)
                loss.backward()
                optimizer.step()

                # display progress bar
                postfix = dict(
                    Loss=f"{loss.item():.3f}",
                    noise=f"{self.model.likelihood.noise.item():.3}",
                )

                record_loss.append(loss.item())
                bar.set_postfix(postfix)
        return record_loss

    def prediction(self, cosmology):
        # Get into evaluation (predictive posterior) mode
        self.model.eval()
        self.likelihood.eval()
        test_x = PreWhiten.x_transformation(self, cosmology).to(torch.float32)

        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            observed_pred = self.likelihood(self.model(test_x))

        return self.yTransform.inverse_tranform(observed_pred.mean)

    def sample(self, cosmology, nsamples=1):
        self.model.eval()
        self.likelihood.eval()
        test_x = PreWhiten.x_transformation(self, cosmology).to(torch.float32)
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            observed_pred = self.likelihood(self.model(test_x))
        samples = torch.normal(
            observed_pred.mean.item(), observed_pred.stddev.item(), size=(nsamples,)
        )
        return self.yTransform.inverse_tranform(samples)
