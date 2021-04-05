import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import MultivariateNormal, Normal


def affine(x, weight, bias):
    # x: stacks of row vectors
    return torch.matmul(x, weight) + bias


def reparameterize(mu, rho):
    epsilon = torch.randn_like(mu)
    sigma = torch.log(1 + torch.exp(rho))
    ret = sigma * epsilon + mu
    return ret


class MixtureGaussian(object):
    def __init__(self, pi, sigma1, sigma2):
        self.pi = pi
        self.simga1 = sigma1
        self.sigma2 = sigma2

        self.normal1 = Normal(0, sigma1)
        self.normal2 = Normal(0, sigma2)

    def log_prob(self, x):
        prob1 = torch.exp(self.normal1.log_prob(x))
        prob2 = torch.exp(self.normal2.log_prob(x))
        ret = torch.log(self.pi * prob1 + (1-self.pi) * prob2).sum()

        return ret


class Gaussian(object):
    def __init__(self, mu, rho):
        self.mu = mu
        self.rho = rho

    @property
    def sigma(self):
        return torch.log(torch.exp(self.rho))

    def sample(self):
        return self.mu + torch.randn_like(self.mu) * self.sigma

    def log_prob(self, x):
        ret = (-math.log(math.sqrt(2*math.pi))
               - torch.log(self.sigma)
               - ((x-self.mu)**2) / (2*self.sigma ** 2)).sum()
        return ret


class Layer(nn.Module):
    def __init__(self, in_features, out_features, pi, sigma_1, sigma_2):
        super(Layer, self).__init__()
        self.weight_mu = nn.Parameter(torch.Tensor(
            in_features, out_features).uniform_(-0.2, 0.2))
        self.weight_rho = nn.Parameter(torch.Tensor(
            in_features, out_features).uniform_(-5, -4))
        self.weight = Gaussian(self.weight_mu, self.weight_rho)
        # Bias parameters
        self.bias_mu = nn.Parameter(
            torch.Tensor(out_features).uniform_(-0.2, 0.2))
        self.bias_rho = nn.Parameter(
            torch.Tensor(out_features).uniform_(-5, -4))
        self.bias = Gaussian(self.bias_mu, self.bias_rho)
        # Prior distributions
        self.weight_prior = MixtureGaussian(pi, sigma_1, sigma_2)
        self.bias_prior = MixtureGaussian(pi, sigma_1, sigma_2)
        self.log_prior = 0
        self.log_variational_posterior = 0

    def forward(self, x):
        weight = self.weight.sample()
        bias = self.bias.sample()
        if self.training:
            self.log_prior = self.weight_prior.log_prob(
                weight) + self.bias_prior.log_prob(bias)
            self.log_variational_posterior = self.weight.log_prob(
                weight) + self.bias.log_prob(bias)
        ret = F.relu(affine(x, weight, bias))
        return ret


class BayesNet(nn.Module):
    def __init__(self, args):
        super(BayesNet, self).__init__()
        l1 = Layer(args.input_size,
                   args.hidden_size,
                   args.pi,
                   args.sigma_1,
                   args.sigma_2)
        layers = [l1]
        for _ in range(args.num_layers - 2):
            layers.append(Layer(args.hidden_size,
                                args.hidden_size,
                                args.pi,
                                args.sigma_1,
                                args.sigma_2))
        layers.append(Layer(args.hidden_size,
                            args.output_size,
                            args.pi,
                            args.sigma_1,
                            args.sigma_2))
        self.layers = nn.Sequential(*layers)

        # for w in self.parameters():
        #     if len(w.size()) > 1:
        #         nn.init.xavier_uniform_(w)
        #     else:
        #         nn.init.zeros_(w)
    
    def forward(self, x):
        logits = self.layers(x)
        return F.log_softmax(logits, dim=-1)

    def log_prior(self):
        log_prob = 0.0
        for layer in self.layers:
            log_prob = log_prob + layer.log_prior
        return log_prob

    def log_variational_posterior(self):
        log_prob = 0.0
        for layer in self.layers:
            log_prob = log_prob + layer.log_variational_posterior
        return log_prob

    def sample_elbo(self, x, target, num_batches, num_samples=1):
        all_q = 0
        all_p = 0
        criterion = nn.NLLLoss(reduction="sum")
        all_logprob = 0
        for _ in range(num_samples):
            log_prob = self(x)
            log_prior = self.log_prior()
            log_variational_posterior = self.log_variational_posterior()
            all_p = all_p + log_prior
            all_q = all_q + log_variational_posterior
            all_logprob = all_logprob + log_prob

        nll = criterion(all_logprob / num_samples, target)
        kl = (all_q - all_p) / num_samples
        loss = kl / num_batches + nll

        return loss, nll, kl
