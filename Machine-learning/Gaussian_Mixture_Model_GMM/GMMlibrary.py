import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D 
from pandas import DataFrame
import pandas as pd
import numpy as np
from numpy.random import randn
import glob
import sys

import warnings
warnings.filterwarnings('ignore')

from scipy.stats import multivariate_normal


class GaussianMixture():
    #Here you will create a refernce to all the parameters which gets substituted against declared variables
    def __init__(self, gaussians: int, n_iters: int, tol: float, seed: int):
        self.gaussians = gaussians
        self.n_iters = n_iters
        self.tol = tol
        self.seed = seed

    def fit(self, X):

        # data's dimensionality and probability vector initialization
        self.n_row, self.n_col = X.shape     
        self.probability = np.zeros((self.n_row, self.gaussians))
        
        #print(self.probability)
        
        ##Below multicommented block can be used if you want to apply GMM on a dataset without Kmeans result
        
       
        # initialize parameters
        np.random.seed(self.seed)
        chosen = np.random.choice(self.n_row, self.gaussians, replace = False)
        #print("Chosen:",chosen)
        self.means = X[chosen]
        #print("Initial Means:",self.means)
        self.weights = np.full(self.gaussians, 1 / self.gaussians)
        #print("Initial weights:",self.weights)
        
        # for np.cov, rowvar = False, 
        # indicates that the rows represents obervation
        shape = self.gaussians, self.n_col, self.n_col
        self.covs = np.full(shape, np.cov(X, rowvar = False))
       # print("Initial Covariance:",self.covs)
        """
        self.means=m
        self.weights=pi
        self.covs=c
        """

        log_likelihood = 0 #Initializing for iteration
        self.converged = False
        self.log_likelihood_trace = []      
        print("...Entering GMM Clustering...\n")
        for i in range(self.n_iters):
            
            log_likelihood_new = self.Estep(X)
            self.Mstep(X)
            

            if  (abs(log_likelihood_new - log_likelihood) <= self.tol):
                
                self.converged = True
                break
  
            log_likelihood = log_likelihood_new
            self.log_likelihood_trace.append(log_likelihood)
            print("Iteration: ",i,"  log_likelihood: ", log_likelihood)
        
        plt.plot(self.log_likelihood_trace)
        plt.title("Loglikelihood Convergence Graph")
        
        
        #print("log_likelihood_trace:",self.log_likelihood_trace)
        last=self.log_likelihood_trace[-1]
        #print(last)

        return self.means,self.weights,self.covs,self.probability

    def Estep(self, X):
        """
        E-step: compute probability,
        update probability matrix so that probability[i, j] is the probability of cluster k 
        for data point i,
        to compute likelihood of data point i belonging to given cluster k, 
        use multivariate_normal.pdf
        """
        self._compute_log_likelihood(X)
        
        self.log_likelihood1 = np.sum(np.log(np.sum(self.probability, axis = 1)))
        
         #Normalization       
        self.probability = self.probability / self.probability.sum(axis = 1, keepdims = 1)
        #print("Normalised probability",self.probability)
        return self.log_likelihood1

    def _compute_log_likelihood(self, X):
        for k in range(self.gaussians):
            
                prior = self.weights[k]
                #print("prior_weight",prior)
                likelihood = multivariate_normal(self.means[k], self.covs[k]).pdf(X)
                #print("Likelihood/probability"+str(k),likelihood)
                self.probability[:, k] = prior * likelihood
                #print(" Size of Initial Probability of all the datapoints in cluster"+str(k),self.probability.shape)          

        return self



    def compute_log_likelihood(self, X):
        self.probs = np.zeros((X.shape[0] , self.gaussians))
        
        for k in range(self.gaussians):

            prior = self.weights[k]            
            #print("prior_weight",prior)
            self.likeli = multivariate_normal(self.means[k], self.covs[k]).pdf(X)
            #print("Likelihood/probability"+str(k),likelihood)

            self.probs[:,k]= prior * self.likeli
            #print(" Size of Initial Probability of all the datapoints in cluster"+str(k),self.probability.shape)       

        self.probs = self.probs / (np.sum(self.probs, axis=1)[:, np.newaxis])
        
        return self.probs



    def compute_log_likelihood_newmean(self, X, nmean, nvar, nweights):
        self.probs1 = np.zeros((X.shape[0], self.gaussians))
        
        for k in range(self.gaussians):

            prior = nweights[k]
            #print("prior_weight",prior)
            self.likeli = multivariate_normal(nmean[k], nvar[k]).pdf(X)
            #print("Likelihood/probability"+str(k),likelihood)

            self.probs1[:,k]= prior * self.likeli
            #print(" Size of Initial Probability of all the datapoints in cluster"+str(k),self.probability.shape)
       

        self.probs1 = self.probs1 / (np.sum(self.probs1, axis=1)[:, np.newaxis])
        
        return self.probs1


    def Mstep(self, X):
        """M-step, update parameters"""

        # total probability assigned to each cluster, Soft alocation(N^soft)
        #print("probability assigned to each cluster",self.probability.sum(axis = 0))
        resp_weights = self.probability.sum(axis = 0)
        
        # updated_weights
        self.weights = resp_weights / X.shape[0]

        # updated_means
        weighted_sum = np.dot(self.probability.T, X)
        self.means = weighted_sum / resp_weights.reshape(-1, 1)
        # updated_covariance
        for k in range(self.gaussians):
            diff = (X - self.means[k]).T
            weighted_sum = np.dot(self.probability[:, k] * diff, diff.T)
            self.covs[k] = weighted_sum / resp_weights[k]
            
        return self
    
    def predict(self, X):
       
        post_proba = np.zeros((X.shape[0], self.gaussians))
        
        for c in range(self.gaussians):
            post_proba [:,c] = self.weights[c] * multivariate_normal.pdf(X, self.means[c,:], self.covs[c])
            #print("Posterior_probability:", post_proba)
        labels  =  post_proba.argmax(1)
        #print("Labels/Classes:",labels)
        
        return labels