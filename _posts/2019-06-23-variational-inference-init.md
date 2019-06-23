---
layout: post_ext
title:  "Why breaking symmetry in variational inference initialisation is needed"
mathjax: true
---

## Introduction

A common task in Bayesian inference is to compute the *marginal or posterior* distribution of a random variable that represents a model parameter by applying Bayes' theorem. Although the theorem itself is simple in concept, computing the marginal distribution can be cumbersome or even numerically not tractable when the number of variables involved, i.e. the complexity of the model, increases. This means that in practice, approximative methods are required to estimate a solution that is sufficiently close to the exact marginal distribution according to a certain measure of distance. Two types of methods are commonly applied: *Monte Carlo sampling* and *variational inference*. Monte-Carlo methods are referred to as "stochastic", because they attempt to find an approximation of the marginal distribution through sequential sampling schemes. In contrast, variational inference methods are referred to as "deterministic" because they are based on a functional approximation, but they also require an iterative algorithm to obtain a solution. Although good expositions can be found for the theoretical foundation of variational inference (see e.g. <span id="a1">[[1]](#Bishop)</span>, <span id="a2">[[2]](#Winn)</span>), the practical requirements for a good initialisation of the algorithm are usually not explained very well. The aim of this post is to explain in more depth one of the initialisation problems that can arise in variational inference based on a Gaussian mixture model.   

## Variational inference in a nutshell

In a nutshell, variational inference aims at approximating the marginal distribution of a set of hidden variables ${V_i}$ (which represent model parameters) given other observed variables ${X_{j}}$ by means of a *variational distribution* $Q({V_i})$:

$$
\begin{equation}
p({V_i} | {X_j} ) \approx Q({V_i})
\end{equation}
$$

where the observed variables are substituted by their observed values and are hence omitted on the right hand side. The goal is then to find the variational distribution by minimising a distance measure between the variational and the true marginal distribution. In variational inference, the [*Kullback-Leibler (KL) divergence*](https://en.wikipedia.org/wiki/Kullback–Leibler_divergence) is chosen as the distance measure. It turns out that by using the KL-divergence, the minimisation equations are mathematically nicer to derive. The form of the variational distribution is typically chosen to allow for simplifications that keep solving the minimisation problem numerically tractable. One way to achieve this is to express the variational distribution as a product of factors that separates the variables ${V_i}$ as if they were *indepedent*:   

$$
\begin{equation}
Q({V_i})  = \prod_{i = 1}^{M} Q_{V_{i}}(V_{i})
\end{equation}
$$

This means that we approximate the marginal distribution by a product of simpler distributions each only dependent on one of the variables. It turns out that this factorization allows to express the KL minimization as a set of *coupled* equations in which the solution for each factor can be expressed explicitly in function of the other factors:

$$
\begin{equation}
log\ Q_{V_i}^{\star}(V_i) = \left\langle log\ p({V_k}, {X_l}) \right\rangle_{\neg Q_{V_i}} \label{eq:factors}
\end{equation}
$$

where $\left\langle . \right\rangle_{\neg Q_{V_i}}$ means computing the expectation with respect to all factors except for $Q_{V_i}$. An advantage of this is that the solution can be obtained by *iteratively* computing an approximation for each factor, while keeping the others constant. Another advantage is that the posterior distributions for each separate variable are also computed in the process. Note that we have made the implicit assumption that the variables are *independent*, and hence we have lost the information about the covariance between the variables as a trade-off. 

An additional simplication can be made by selecting [*conjugate*](https://en.wikipedia.org/wiki/Conjugate_prior) *models* for the joint probability $p({V_k}, {X_l})$. In this case, the factors $Q_{V_i}$ can be selected to have the same type of distributions as the priors of the hidden variables ${V_i}$. This means in practice, to find the factors, we can express equation (\ref{eq:factors}) in terms of the distribution *parameters*, such that they become algebraic equations. This will be illustrated by the example of the Gaussian mixture model.  

## Variational inference of Gaussian mixture model

To get a better intuition for how variational inference works in practice, it is educational to derive the variational inference equations for a one-dimensional Gaussian mixture model. The mathematics are a bit elaborate to write down, but because the distributions are from the exponential family and we can leverage the conjugacy constraint, the computation is rather straight-forward. 

A Gaussian mixture is a distribution that consists of a weighted sum of $K$ Gaussian distributions. The weights indicate the probability of selecting a component when drawing samples. Therefore, sampling from a Gaussian mixture happens in two stages: sampling one of the $K$ components and sampling from the Gaussian distribution associated with the sampled component. Both sampling stages are independent from each other. For a more detailed exposition, the reader is referred to resources such as <span id="a3">[[1]](#Bishop)</span>. In the following, the notation $\\{\cdot\\}$ indicates a set of random variables. For a Gaussian mixture model, the *likelihood* $\mathcal{L}$ of the observations $\\{x_i\\}$ is given by:

$$
\begin{equation}
\mathcal{L} = p(\{x_i\}| \{\mathbf{z}_i\}, \{\mu_k\}, \{\gamma_k\}) = \prod_{i = 1}^{N} \prod_{k = 1}^{K} \pi_k^{z_{ik}}\mathcal{N}(x_i|\mu_k, \gamma_k)^{z_{ik}}
\end{equation}
$$

where $\mathcal{N}$ represents a Gaussian distribution and $\mu_k$, $\gamma_k$ and $\pi_k$ are respectively the mean, precision and weight of the kth component, with $\sum_{k = 1}^{K}\pi_{k} = 1$. The *latent or selector* variables $\mathbf{z}\_i = (z_{i1}, \cdots, z_{iK})$ select the component that the ith observation $x_i$ belongs to, using a categorical representation, i.e. $z_{ik}$ is binary and $\sum_{k = 1}^{K}z_{ik} = 1$. Hence, the variables $\mathbf{z}\_i$ are governed by the discrete distribution $p(z_{ik} = 1 \| \\{\pi_k\\}) = \pi_k$ which can also be expressed as $p(\mathbf{z}\_i \| \\{\pi_k\\}) = \prod_{k = 1}^{K}\pi_{k}^{z_{ik}}$. Typical conjugate priors for $\mu_k$, $\gamma_k$, $\pi_k$ are:

$$
\begin{equation}
\begin{aligned}
Pr(\mu_k) & =  \mathcal{N}(\mu_k|m, \beta) \\
Pr(\gamma_k) & = G(\gamma_k|a, b) \\
Pr(\{\pi_k\}) & = D(\{\pi_k\}|\{u_k\}) 
\end{aligned}
\end{equation}
$$

with $G$ and $D$ representing the Gamma and Dirichlet distributions respectively. Note that all $\mu_k$ and $\gamma_k$ share the same prior-distribution. The variational distribution to approximate the posterior distribution can then be constructed as:

$$
\begin{equation}
\begin{aligned}
Q & = Q(\{\mathbf{z}_i\}, \{\mu_k\}, \{\gamma_k\}, \{\pi_k\}) \\
& = Q_{\{\pi_k\}}\prod_{k = 1}^{K}Q_{\mu_k}Q_{\gamma_k}\prod_{i = 1}^{N}Q_{\mathbf{z}_i}
\end{aligned}
\end{equation}
$$

Because of the conjugacy constraint, the factors will be distributions with the same form as the corresponding priors:

$$
\begin{equation}
\begin{aligned}
Q_{\mu_k}(\mu_k) & =  \mathcal{N}(\mu_k|m_k', \beta_k') \\
Q_{\gamma_k}(\gamma_k) & = G(\gamma_k|a_k', b_k') \\
Q_{\{\pi_k\}}(\{\pi_k\}) & = D(\{\pi_k\}|\{u_k'\}) \\
Q_{\mathbf{z}_i}(\{z_{ik}\}) & = \text{Discrete distribution with k values} 
\end{aligned} \label{eq:factors_mixture}
\end{equation}
$$

To compute the factors in the variational distribution we need to construct the joint probability $\mathcal{P}$:

$$
\begin{equation}
\begin{aligned}
\mathcal{P} & = p(\{X_i\}, \{\mathbf{z}\_i\}, \{\mu_k\}, \{\gamma_k\}, \{\pi_k\}) \\
& = \mathcal{L}Pr(\{\pi_k\})\prod_{k = 1}^{k} \left[Pr(\mu_k) Pr(\gamma_k)\right] 
\end{aligned}
\end{equation}
$$

The Gaussian, Gamma and Dirichlet distributions are all from the exponential family and by using their general formulation, we can write $log\,\mathcal{P}$ as: 

$$
\begin{equation}
\begin{aligned}
log\,\mathcal{P} &= \sum_{i = 1}^{N} \sum_{k = 1}^{K} z_{ik}\left[log\,\pi_k + \gamma_k\mu_k x_i - \frac{\gamma_k}{2}x_i^2 + \frac{1}{2}\left( log\,\gamma_k - \gamma_k\mu_k^2 - log\,2\pi \right)\right] \\
 &  + \sum_{k = 1}^{K} \left[ \beta m \mu_k - \frac{\beta}{2}\mu_k^{2} + \frac{1}{2}\left( log\,\beta - \beta m^2 - log\,2\pi \right) \right] \\
 &  + \sum_{k = 1}^{K} \left[ (a - 1)log\,\gamma_k - b\gamma_k + a\,log\,b - log\,\Gamma(a) \right] \\
 &  + \sum_{k = 1}^{K}(u_{k}-1)log\,\pi_k + log\,\Gamma\left( \sum_{k = 1}^{K} u_k \right) - \sum_{k = 1}^{K} \Gamma(\mu_k)
\end{aligned} \label{eq:joint_prob_mixture}
\end{equation}
$$

where $\Gamma$ represents the Gamma-function. Because the variational factors are also in the exponential family, we can compute their distribution parameters (the primed parameters in the equations (\ref{eq:factors_mixture})) as *function of the expectations* of the hidden variables by applying equation (\ref{eq:factors}). We can achieve this by considering the following:

* The expectation operator is *distributive* with respect to the sum operation.
* Because of the factorization, we can assume that the variables are *independent* (which doesn't mean that they are, it's just how we approximated them). This implies that to compute the expectation of a distribution parameter, we only need to do so with respect to the associated factor.
* For each factor, we only need to take into consideration the terms in equation (\ref{eq:joint_prob_mixture}) that contain the variable of the factor. The other terms will evaluate to constant terms under the expectation operator and can hence be aggregated in the normalization constants (which we don't need to compute explicitly because we consider parameterized distribution, with the exception of $Q_{\mathbf{z}\_i}$)

The factors for the latent variables $\mathbf{z}\_i$ are a bit special, since their distributions are discrete and hence are not parametrized. For these factors, we need to compute the probability of each component $z_{ik}$. We'll start with computing these by applying equation (\ref{eq:factors}) with the log of the joint probability given by equation (\ref{eq:joint_prob_mixture}), and we only need to consider the terms with $z_{ik}$ which gives:

$$
\begin{equation}
log\ Q_{\mathbf{z}_i} = \sum_{k = 1}^{K} z_{ik} \left[\langle log\,\pi_k \rangle_{Q_{\{\pi_k\}}} + 
\left\langle \mathcal{N}(x_i|\mu_k, \gamma_k) \right\rangle_{Q_{\mu_k}Q_{\gamma_k}} \right] + \text{const}
\end{equation}
$$

Note that $Q_{\mathbf{z}\_i}$ is supposed to approximate the posterior distribution for $\mathbf{z}\_i$ given the observation $x_i$. This is reflected in the dependency not only on the mixture weights $\\{\pi_{k}\\}$ but also on the expected likelihood of the observation $x_i$ for each Gaussian component. If we work this out further, then we obtain:

$$
\begin{equation}
\begin{aligned}
log\ Q_{\mathbf{z}_i} &= \sum_{k = 1}^{K} z_{ik} \left[\langle log\,\pi_k \rangle_{Q_{\{\pi_k\}}} + \frac{1}{2}\langle log\,\gamma_k \rangle_{Q_{\gamma_k}} + \right. \\
& + \left. \langle \gamma_k \rangle_{Q_{\gamma_k}} \left( x_{i}\langle \mu_k \rangle_{Q_{\mu_k}} - \frac{1}{2}x_i^2 - 
\frac{1}{2}\langle \mu_k^2 \rangle_{Q_{\mu_k}} \right) - \frac{1}{2}log\,2\pi \right] + \text{const} \\
&= \sum_{k = 1}^{K} z_{ik}\theta_{ik} + \text{const} 
\end{aligned}
\end{equation}
$$

Since we know that $Q_{\mathbf{z}\_i}$ should be a discrete distribution, the normalization factor or *partition function* can be computed as $\sum_{k = 1}^{K}\theta_{ik}$ and we obtain that:

$$
\begin{equation}
\begin{aligned}
Q_{\mathbf{z}_i} & = \prod_{k=1}^{K}\left(\frac{\theta_{ik}}{\sum_{k = 1}^{K}\theta_{ik}}\right)^{z_{ik}} \\
& = \prod_{k=1}^{K}Q_{\mathbf{z}_i}(k)^{z_{ik}}
\end{aligned}
\end{equation}
$$

Since the expectation $\langle z_{ik} \rangle_{Q_{\mathbf{z}\_i}}$ is given by $Q_{\mathbf{z}\_i}(k)$, we can now compute the distribution parameters for $Q_{\mu_k}$ by applying equation (\ref{eq:factors}):

$$
\begin{equation}
log\ Q_{\mu_k} = \sum_{i = 1}^{N} Q_{\mathbf{z}_i}(k) \left[ \mu_{k}x_{i}\langle \gamma_k \rangle_{Q_{\gamma_k}}
- \frac{1}{2}\mu_k^2 \langle \gamma_k \rangle_{Q_{\gamma_k}} \right] + m \beta \mu_{k} - \frac{1}{2}\beta\mu_k^2 + \text{const} \label{eq:factor_mu_expect}
\end{equation}
$$

Since the factor $Q_{\mu_k}$ is Gaussian because the prior is Gaussian, we know that it can be expressed in the same general form:

$$
\begin{equation}
log\ Q_{\mu_k} = m_k'\beta_k'\mu_k - \frac{1}{2}\beta_k'\mu_k^2 + \cdots \label{eq:factor_mu}
\end{equation}
$$

Matching the terms with $\mu_k$ and $\mu_k^2$ between equations (\ref{eq:factor_mu_expect}) and (\ref{eq:factor_mu}) then gives:

$$
\begin{equation}
\begin{aligned}
\beta_k' & = \langle \gamma_k \rangle_{Q_{\gamma_k}}\sum_{i = 1}^{N}Q_{\mathbf{z}_i}(k) + \beta \\
m_k' & = \frac{1}{\beta_k'}\left( \langle \gamma_k \rangle_{Q_{\gamma_k}} \sum_{i = 1}^{N}x_{i}Q_{\mathbf{z}_i}(k) + m \beta \right)
\end{aligned}
\end{equation}
$$

We can proceed in the same way to compute the distribution parameters for $Q_{\gamma_k}$. Applying equation (\ref{eq:factors}) gives:

$$
\begin{equation}
\begin{aligned}
log\ Q_{\gamma_k} & = \sum_{i = 1}^{N} Q_{\mathbf{z}_i}(k) \left[ \gamma_{k}x_{i}\langle \mu_k \rangle_{Q_{\mu_k}} +
\frac{1}{2}\left(log\,\gamma_k - x_i^2\gamma_k - \gamma_k\langle \mu_k^2 \rangle_{Q_{\mu_k}} \right) \right] \\
& - b\gamma_k + (a - 1)log\,\gamma_k + \text{const}
\end{aligned}
\end{equation}
$$

The factor $Q_{\gamma_k}$ is a Gamma distribution and matching the terms with $\gamma_k$ and $log\,\gamma_k$ gives the parameters:

$$
\begin{equation}
\begin{aligned}
a_k' &= \frac{1}{2}\sum_{i = 1}^{N} Q_{\mathbf{z}_i}(k) + a \\
b_k' &= \sum_{i = 1}^{N} Q_{\mathbf{z}_i}(k)\left[ \frac{x_i^2}{2} - x_i\langle \mu_k \rangle_{Q_{\mu_k}} 
+ \frac{1}{2}\langle \mu_k^2 \rangle_{Q_{\mu_k}} \right] + b  
\end{aligned}
\end{equation}
$$

Finally, for $Q_{\\{ \pi_k \\}}$, applying equation (\ref{eq:factors}) gives:

$$
\begin{equation}
log\ Q_{\{ \pi_k \}} = \sum_{k = 1}^{K}\left( \sum_{i = 1}^{N}Q_{\mathbf{z}_i}(k) + u_k - 1 \right)log\,\pi_k + \text{const}
\end{equation}
$$

and matching the Dirichlet distribution parameters then gives:

$$
\begin{equation}
u_k' = \sum_{i = 1}^{N}Q_{\mathbf{z}_i}(k) + u_k
\end{equation}
$$

## Initialisation: breaking the symmetry

In summary, variational inference of a one-dimensional Gaussian mixture model involves the following set of equations:

$$
\begin{align}
\theta_{ik} &= \langle log\,\pi_k \rangle_{Q_{\{\pi_k\}}} + \frac{1}{2}\langle log\,\gamma_k \rangle_{Q_{\gamma_k}} + \nonumber \\
& + \langle \gamma_k \rangle_{Q_{\gamma_k}} \left( x_{i}\langle \mu_k \rangle_{Q_{\mu_k}} - \frac{1}{2}x_i^2 -
\frac{1}{2}\langle \mu_k^2 \rangle_{Q_{\mu_k}} \right) - \frac{1}{2}log\,2\pi \label{eq:theta} \\
\color{green}{Q_{\mathbf{z}_i}(k)} &= \frac{\theta_{ik}}{\sum_{k = 1}^{K}\theta_{ik}} \\
\beta_k' &= \langle \gamma_k \rangle_{Q_{\gamma_k}} \color{green}{\sum_{i = 1}^{N}Q_{\mathbf{z}_i}(k)} + \beta \label{eq:param_first} \\
m_k' & = \frac{1}{\beta_k'}\left( \langle \gamma_k \rangle_{Q_{\gamma_k}} \color{green}{\sum_{i = 1}^{N}}x_{i}\color{green}{Q_{\mathbf{z}_i}(k)} + m \beta \right) \\
a_k' &= \frac{1}{2}\color{green}{\sum_{i = 1}^{N} Q_{\mathbf{z}_i}(k)} + a \\
b_k' &= \color{green}{\sum_{i = 1}^{N} Q_{\mathbf{z}_i}(k)}\left[ \frac{x_i^2}{2} - x_i\langle \mu_k \rangle_{Q_{\mu_k}} 
+ \frac{1}{2}\langle \mu_k^2 \rangle_{Q_{\mu_k}} \right] + b \\
u_k' &= \color{green}{\sum_{i = 1}^{N}Q_{\mathbf{z}_i}(k)} + u_k \label{eq:param_last}
\end{align} 

$$

Since these equations are coupled, they require an iterative approach to be solved. This is done by first initialising all but one of the factors, and each factor is then updated consecutively by computing the associated distribution parameters while keeping the other parameters constant. According to <span id="a2">[[2]](#Winn)</span>, the typical way for initialising variational inference is to set all factors to be *broad* distributions. In the case of the Gaussian mixture model, this would mean that we would initialise all $Q_{\mathbf{z}\_i}$ to be *uniform* discrete distributions with $Q_{\mathbf{z}\_i}(k) = 1/K$, and consequently, according to equation (\ref{eq:theta}), the distribution parameters of the factors for all $K$ components would be initialised with the same value. However, as can be seen in the equations (\ref{eq:param_first}) - (\ref{eq:param_last}), all parameters are updated using a sum of $\color{green}{Q_{\mathbf{z}\_i}(k)}$ over all $N$ observations, which has the same value for all $k$ since $Q_{\mathbf{z}\_i}$ is uniform. But this would mean that for each component $k$, the parameters would be *updated with the same value*, such that $Q_{\mathbf{z}\_i}$ would stay uniform. With this initialisation, we cannot distinguish the $K$ components in the Gaussian mixture model as being different and the algorithm would "converge" to a solution in which all components are the same. One way to *break the symmetry* is to perturb the initialisation of $Q_{\mathbf{z}\_i}$ such that it is not uniform for all observations. This can be done e.g. by randomly selecting a component $j$ for each observation such that $Q_{\mathbf{z}\_i}(j) = 1$, as also mentioned in the [Infer.Net tutorial](https://dotnet.github.io/infer/userguide/Mixture%20of%20Gaussians%20tutorial.html). You can also have a look at [my github repository](https://github.com/Tobimaru/InferNet-pythonnet) which contains an implementation of this Infer.Net example using a python wrapper in a Jupyter notebook.


## References

1. <span id="Bishop"></span> C.M. Bishop, [*Pattern recognition and machine learning*](https://www.microsoft.com/en-us/research/publication/pattern-recognition-machine-learning/), Springer, 2006.

2. <span id="Winn"></span> J.M. Winn, [*Variational Message Passing and its
Applications*](http://www.johnwinn.org/Publications/thesis/Winn03_thesis.pdf), PhD-thesis, 2003.

 
 




