import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
from sklearn.datasets.samples_generator import make_blobs
from scipy.stats import multivariate_normal

class GMM():
    def __init__(self, k, max_iter=5):
        self.k = k
        self.max_iter = int(max_iter)

    def initialize(self, X):
        self.shape = X.shape
        self.n, self.m = self.shape

        self.phi = np.full(shape=self.k, fill_value=1/self.k)
        self.weights = np.full( shape=self.shape, fill_value=1/self.k)
        
        random_row = np.random.randint(low=0, high=self.n, size=self.k)
        self.mu = [  X[row_index,:] for row_index in random_row ]
        self.sigma = [ np.cov(X.T) for _ in range(self.k) ]

    def e_step(self, X):
        # E-Step: update weights and phi holding mu and sigma constant
        self.weights = self.predict_proba(X)
        self.phi = self.weights.mean(axis=0)
    
    def m_step(self, X):
        # M-Step: update mu and sigma holding phi and weights constant
        for i in range(self.k):
            weight = self.weights[:, [i]]
            total_weight = weight.sum()
            self.mu[i] = (X * weight).sum(axis=0) / total_weight
            self.sigma[i] = np.cov(X.T, 
                aweights=(weight/total_weight).flatten(), 
                bias=True)

    def fit(self, X):
        self.initialize(X)
        log = []
        for iteration in range(self.max_iter):

            self.e_step(X)
            self.m_step(X)

            log.append(np.array(sumoflikelihood).sum())
            error = log[iteration] - log[iteration-1];
            print(f'Likelihood for each {iteration} iteration',np.array(sumoflikelihood).sum())
            if iteration == 0:
                pass;
            elif error<0.00001:
                break;
            fig = plt.figure()
            classes = self.predict(X)
            ax = fig.add_subplot(111)
            # plot x,y data with c as the color vector, set the line width of the markers to 0
            ax.scatter(X[:,0], X[:,1], c=classes, lw=0)
            x,y = np.meshgrid(np.sort(X[:,0]),np.sort(X[:,1]))
            XY = np.array([x.flatten(),y.flatten()]).T
            for m,c in zip(self.mu,self.sigma):
                multi_normal = multivariate_normal(mean=m,cov=c)
                ax.contour(np.sort(X[:,0]),np.sort(X[:,1]),multi_normal.pdf(XY).reshape(len(X),len(X)),colors='black',alpha=0.3)
                ax.scatter(m[0],m[1],c='grey',zorder=10,s=100)
            


        fig1 = plt.figure()
        ax1 = fig1.add_subplot(111)
        ax1.set_title('Likelihood for each iteration')
        ax1.plot(log)
        plt.show()

        return (self, self.mu, self.sigma, self.weights, self.phi)
        
        


            
    def predict_proba(self, X):
        

        
        global sumoflikelihood
        sumoflikelihood = []
        likelihood = np.zeros( (self.n, self.k) )
        for i in range(self.k):
            distribution = multivariate_normal(
                mean=self.mu[i], 
                cov=self.sigma[i])
            likelihood[:,i] = distribution.pdf(X)
        sumoflikelihood.append(likelihood.sum())    
        
        numerator = likelihood * self.phi
        denominator = numerator.sum(axis=1)[:, np.newaxis]
        weights = numerator / denominator
        return weights
    
    def predict(self, X):
        weights = self.predict_proba(X)
        return np.argmax(weights, axis=1)
