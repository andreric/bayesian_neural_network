# -*- coding: utf-8 -*-
# @Author: Andre Goncalves
# @Date:   2019-10-31 16:27:32
# @Last Modified by:   Andre Goncalves
# @Last Modified time: 2019-11-15 14:01:38

""" code from: https://joshfeldman.net/ml/2018/12/17/WeightUncertainty.html
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt


class Linear_BBB(nn.Module):
    """
        Layer of our BNN.
    """

    def __init__(self, input_features, output_features, prior_std=1.):
        """
            Initialization of our layer : our prior is a normal distribution
            centered in 0 and of variance 20.
        """
        # initialize layers
        super().__init__()
        # set input and output dimensions
        self.input_features = input_features
        self.output_features = output_features

        # initialize mu and rho parameters for the weights of the layer
        self.w_mu = nn.Parameter(torch.zeros(output_features, input_features))
        self.w_rho = nn.Parameter(torch.zeros(output_features, input_features))

        # initialize mu and rho parameters for the layer's bias
        self.b_mu = nn.Parameter(torch.zeros(output_features))
        self.b_rho = nn.Parameter(torch.zeros(output_features))

        # initialize weight samples (these will be calculated whenever the layer makes a prediction)
        self.w = None
        self.b = None

        # initialize prior distribution for all of the weights and biases
        self.prior = torch.distributions.Normal(0, prior_std)

    def forward(self, input):
        """
          Optimization process
        """
        # sample weights
        w_epsilon = Normal(0, 1).sample(self.w_mu.shape)
        self.w = self.w_mu + torch.log(1 + torch.exp(self.w_rho)) * w_epsilon

        # sample bias
        b_epsilon = Normal(0, 1).sample(self.b_mu.shape)
        self.b = self.b_mu + torch.log(1 + torch.exp(self.b_rho)) * b_epsilon

        # record log prior by evaluating log pdf of prior at sampled weight and bias
        w_log_prior = self.prior.log_prob(self.w)
        b_log_prior = self.prior.log_prob(self.b)
        self.log_prior = torch.sum(w_log_prior) + torch.sum(b_log_prior)

        # record log variational posterior by evaluating log pdf of normal distribution defined by parameters with respect at the sampled values
        self.w_post = Normal(self.w_mu.data, torch.log(1 + torch.exp(self.w_rho)))
        self.b_post = Normal(self.b_mu.data, torch.log(1 + torch.exp(self.b_rho)))
        self.log_post = self.w_post.log_prob(self.w).sum() + self.b_post.log_prob(self.b).sum()

        return F.linear(input, self.w, self.b)


class MLP_BBB(nn.Module):

    def __init__(self, input_size, output_size,
                 hidden_units, activ_func, noise_tol=.1, prior_std=1.):

        self.arch = hidden_units
        self.activ_func = activ_func
        # initialize the network like you would with a standard multilayer perceptron, but using the BBB layer
        super().__init__()

        self.layers = nn.ModuleList([Linear_BBB(input_size, self.arch[0], prior_std=prior_std)])  # input layer
        for i in range(1, len(hidden_units)):
            self.layers.extend([Linear_BBB(self.arch[i - 1], self.arch[i], prior_std=prior_std)])  # hidden layers
        self.layers.append(Linear_BBB(self.arch[-1], output_size, prior_std=prior_std))  # output layers

        self.noise_tol = noise_tol  # we will use the noise tolerance to calculate our likelihood

    def forward(self, x):

        # standard multilayer perceptron
        for i in range(len(self.layers) - 1):
            if self.activ_func[i] == 'relu':
                x = torch.relu(self.layers[i](x))
            elif self.activ_func[i] == 'tanh':
                x = torch.tanh(self.layers[i](x))
            else:
                raise ValueError("Unknown activation function: {}".format(self.activ_func[i]))
        x = self.layers[-1](x)
        return x

    def log_prior(self):
        # calculate the log prior over all the layers
        log_prior = 0
        for i in range(len(self.layers)):
            log_prior += self.layers[i].log_prior
        return log_prior

    def log_post(self):
        log_post = 0
        for i in range(len(self.layers)):
            log_post += self.layers[i].log_post
        # calculate the log posterior over all the layers
        return log_post  # self.hidden.log_post + self.out.log_post

    def sample_elbo(self, input, target, samples):
        # we calculate the negative elbo, which will be our loss function
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


# def test():

#     net = MLP_BBB(32, prior_std=10)
#     optimizer = optim.Adam(net.parameters(), lr=.1)
#     epochs = 2000
#     for epoch in range(epochs):  # loop over the dataset multiple times
#         optimizer.zero_grad()
#         # forward + backward + optimize
#         loss = net.sample_elbo(x, y, 1)
#         loss.backward()
#         optimizer.step()
#         if epoch % 10 == 0:
#             print('epoch: {}/{}'.format(epoch + 1, epochs))
#             print('Loss:', loss.item())
#     print('Finished Training')
