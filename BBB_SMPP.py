# -*- coding: utf-8 -*-
# @Author: Andre Goncalves
# @Date:   2019-10-31 16:27:32
# @Last Modified by:   Andre Goncalves
# @Last Modified time: 2019-11-15 13:37:53

""" code from: https://joshfeldman.net/ml/2018/12/17/WeightUncertainty.html """
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


# class Gaussian(object):
#     def __init__(self, mu, rho):
#         super().__init__()
#         self.mu = mu
#         self.rho = rho
#         self.normal = torch.distributions.Normal(0, 1)

#     @property
#     def sigma(self):
#         return torch.log1p(torch.exp(self.rho))

#     def sample(self):
#         epsilon = self.normal.sample(self.rho.size())  # .to(DEVICE)
#         return self.mu + self.sigma * epsilon

#     def log_prob(self, input):
#         return (-math.log(math.sqrt(2 * math.pi)) -
#                 torch.log(self.sigma) -
#                 ((input - self.mu) ** 2) / (2 * self.sigma ** 2)).sum()


class ScaleMixtureGaussian_Prior(object):

    def __init__(self, pi, sigma1, sigma2):
        super().__init__()
        self.pi = pi
        self.sigma1 = sigma1
        self.sigma2 = sigma2
        self.gaussian1 = torch.distributions.Normal(0, sigma1)
        self.gaussian2 = torch.distributions.Normal(0, sigma2)

    def log_prob(self, input):
        prob1 = torch.exp(self.gaussian1.log_prob(input))
        prob2 = torch.exp(self.gaussian2.log_prob(input))
        return (torch.log(self.pi * prob1 + (1 - self.pi) * prob2)).sum()


class ScaleMixtureGaussian_VarPost(object):

    def __init__(self, pi, sigma1, sigma2):
        super().__init__()
        self.pi = pi
        self.rho1 = sigma1
        self.rho2 = sigma2

        self.normal = torch.distributions.Normal(0, 1)
        self.gumbel = torch.distributions.Gumbel(0, 1)

        self.gaussian1 = torch.distributions.Normal(0, self.sigma1)
        self.gaussian2 = torch.distributions.Normal(0, self.sigma2)

    @property
    def sigma1(self):
        return torch.log1p(torch.exp(self.rho1))

    @property
    def sigma2(self):
        return torch.log1p(torch.exp(self.rho2))

    # def sample(self):
    #     """ Sample from variation posterior distribution. """
    #     # print('pi: {}'.format(self.pi.shape))
    #     z = self.gumbel.sample((*self.pi.shape, 2)).squeeze_()
    #     # print('z: {}'.format(z))
    #     if len(self.pi.shape) > 1:
    #         x = torch.stack((self.pi, 1 - self.pi), dim=2).squeeze_()
    #     else:
    #         x = torch.cat((self.pi.view(-1, 1), 1 - self.pi.view(-1, 1)), 1)

    #     # print('x.shape: {}'.format(x.shape))
    #     # print('z.shape: {}'.format(z.shape))

    #     mode = torch.argmax(x + z, dim=1).float().resize_(self.pi.shape)
    #     # print('Mode from gumbel: {}'.format(mode))

    #     epsilon = self.normal.sample(self.rho1.size())
    #     # print('epsilon: {}'.format(epsilon.shape))
    #     # print('sigma1: {}'.format(self.sigma1.shape))
    #     # print('mode: {}'.format(mode.shape))
    #     # print(mode * self.sigma1 * epsilon)
    #     value = mode * self.sigma1 * epsilon + (1 - mode) * self.sigma2 * epsilon
    #     # value.squeeze_(-1)
    #     # print('value: {}'.format(value.shape))
    #     assert self.pi.shape == value.shape
    #     return value
    #     # else:
    #     # return self.sigma2 * epsilon

    def sample(self):
        """ Sample from variation posterior distribution. """

        z = self.gumbel.sample((*self.pi.shape, 2)).squeeze_()
        if len(self.pi.shape) > 1:
            x = torch.stack((self.pi, 1 - self.pi), dim=2).squeeze_()
        else:
            x = torch.cat((self.pi.view(-1, 1), 1 - self.pi.view(-1, 1)), 1)

        mode = torch.argmax(x + z, dim=1).float().resize_(self.pi.shape)

        epsilon = self.normal.sample(self.rho1.size())
        return mode * self.sigma1 * epsilon + (1 - mode) * self.sigma2 * epsilon

    def log_prob(self, input):
        prob1 = torch.exp(self.gaussian1.log_prob(input))
        prob2 = torch.exp(self.gaussian2.log_prob(input))
        return (torch.log(self.pi * prob1 + (1 - self.pi) * prob2)).sum()


class BayesianLinear(nn.Module):
    """ Bayesian layer. """

    def __init__(self, in_features, out_features, priors):

        super().__init__()

        # set input and output dimensions
        self.in_features = in_features
        self.out_features = out_features

        # initialize mu and rho parameters for the weights of the layer
        self.w_pi = nn.Parameter(torch.zeros(self.out_features, self.in_features))
        self.w_rho1 = nn.Parameter(torch.zeros(self.out_features, self.in_features))
        self.w_rho2 = nn.Parameter(torch.zeros(self.out_features, self.in_features))

        self.b_pi = nn.Parameter(torch.zeros(self.out_features))
        self.b_rho1 = nn.Parameter(torch.zeros(self.out_features))
        self.b_rho2 = nn.Parameter(torch.zeros(self.out_features))

        # initialize mu and rho parameters for the layer's bias
        # self.b_mu = nn.Parameter(torch.zeros(self.out_features))
        # self.b_rho = nn.Parameter(torch.zeros(self.out_features))

        # Weight and bias parameters of the variational posterior
        self.weight = ScaleMixtureGaussian_VarPost(self.w_pi, self.w_rho1, self.w_rho2)
        self.bias = ScaleMixtureGaussian_VarPost(self.b_pi, self.b_rho1, self.b_rho2)

        # initialize prior distribution for all of the weights and biases
        self.prior = ScaleMixtureGaussian_Prior(priors['pi'], priors['sigma1'], priors['sigma2'])

        self.log_prior = 0
        self.log_post = 0  # variational posterior

    def forward(self, input, sample=False,
                calculate_log_probs=False):
        """ Stochastic forward. Sample a model and run forward. """

        # if self.training or sample:
        weight = self.weight.sample()
        bias = self.bias.sample()
        # else:
        # weight = self.weight.mu
        # bias = self.bias.mu

        if self.training or calculate_log_probs:
            self.log_prior = self.prior.log_prob(weight) + self.prior.log_prob(bias)
            self.log_post = self.weight.log_prob(weight) + self.bias.log_prob(bias)
        else:
            self.log_prior, self.log_post = 0, 0

        return F.linear(input, weight, bias)


class MLP_BBB(nn.Module):

    def __init__(self, input_size, output_size,
                 hidden_units, prior_pi, prior_sigma1,
                 prior_sigma2, noise_tol=.1):

        self.arch = hidden_units
        self.prior_pi = prior_pi
        self.prior_sigma1 = prior_sigma1
        self.prior_sigma2 = prior_sigma2

        priors = {'pi': self.prior_pi,
                  'sigma1': self.prior_sigma1,
                  'sigma2': self.prior_sigma2}

        # initialize the network like you would with a standard multilayer perceptron, but using the BBB layer
        super().__init__()

        self.layers = nn.ModuleList([BayesianLinear(input_size, self.arch[0], priors)])  # input layer
        for i in range(1, len(hidden_units)):
            self.layers.extend([BayesianLinear(self.arch[i - 1], self.arch[i], priors)])  # hidden layers
        self.layers.append(BayesianLinear(self.arch[-1], output_size, priors))  # output layers

        self.noise_tol = noise_tol  # we will use the noise tolerance to calculate our likelihood

    def forward(self, x):
        # again, this is equivalent to a standard multilayer perceptron
        for i in range(len(self.layers) - 1):
            x = torch.tanh(self.layers[i](x, sample=True))
        x = self.layers[-1](x, sample=True)
        return x

    def log_prior(self):
        # calculate the log prior over all the layers
        log_prior = 0
        for i in range(len(self.layers)):
            log_prior += self.layers[i].log_prior
        return log_prior

    def log_post(self):
        # calculate the log posterior over all the layers
        log_post = 0
        for i in range(len(self.layers)):
            log_post += self.layers[i].log_post
        return log_post

    def sample_elbo(self, input, target, samples):
        """ Calculate the negative elbo, which will be our loss function. """

        # initialize tensors
        outputs = torch.zeros(samples, target.shape[0])
        log_priors = torch.zeros(samples)
        log_posts = torch.zeros(samples)
        log_likes = torch.zeros(samples)

        # make predictions and calculate prior, posterior, and likelihood for a given number of samples
        for i in range(samples):
            outputs[i] = self(input).reshape(-1)  # make predictions
            log_priors[i] = self.log_prior()  # get log prior
            log_posts[i] = self.log_post()  # get log variational posterior
            log_likes[i] = Normal(outputs[i], self.noise_tol).log_prob(target.reshape(-1)).sum()  # calculate the log likelihood

        # calculate monte carlo estimate of prior posterior and likelihood
        log_prior = log_priors.mean()
        log_post = log_posts.mean()
        log_like = log_likes.mean()

        # calculate the negative elbo (which is our loss function)
        loss = log_post - log_prior - log_like

        return loss
