import matplotlib.pyplot as plt
import numpy as np
from numpy import arange, polyfit, array, newaxis
from numpy.core.umath import pi, sqrt, e, log, exp
from numpy.ma import sin, dot
from numpy.random.mtrand import randn, random_sample
from sklearn import linear_model
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures


def polynomial(w, x):
    powers = range(0,len(w))
    powers.reverse()

    powered = array(x)[:,newaxis] ** array(powers)
    return dot(powered, w)

# Define our error function
def err(w, x, t):
    return 0.5 * sum((polynomial(w,x)-t)**2)

def rms(w, x, t):
    return sqrt(2 * err(w, x, t)/len(x))

N = 10

# Degree of polynomial
M = 4

xlab = arange(0, 1, 0.01)
ylab = sin(2 * pi * xlab)

x = arange(0, 1, 1. / N)
tlab = sin(2 * pi * x)


fig, ax = plt.subplots(nrows = 2, ncols = 2)
plt.ylim(-1.5, 1.5)

mDegrees = [1, 2, 3, 9]

std_deviation = 0.3
noise = std_deviation * randn(N)
t = tlab + noise

for idx, val in enumerate(mDegrees):
    w = polyfit(x, t, val)
    row = idx / 2
    col = idx % 2
    fig = ax[row, col]
    fig.set_title('M=%d' % val)
    fig.plot(xlab, ylab)
    fig.plot(x, t, 'ro')
    fig.plot(xlab, polynomial(w, xlab), 'g')



NTest = 8
xTest = random_sample(NTest)
yTest = sin(2 * pi * xTest) + randn(NTest) * std_deviation


test_err = []
train_err = []

maxOrder = 10

figError = plt.figure()

for m in range(0, maxOrder):
    weights = polyfit(x, t, m)
    train_err.append(rms(weights, x, t))
    test_err.append(rms(weights, xTest, yTest))

plt.xlabel("M")
plt.ylabel("RMS ERROR")
plt.plot(range(0, maxOrder), train_err, 'bo-')
plt.plot(range(0, maxOrder), test_err, 'ro-')


fig2, ax = plt.subplots(nrows = 1, ncols = 2)

trainNumbers = [15, 100]
for idx, val in enumerate(trainNumbers):
    std_deviation = 0.3
    trainNoise = std_deviation * randn(val)
    trainX = arange(0, 1, 1. / val)
    trainTlab = sin(2 * pi * trainX )
    trainT = trainTlab + trainNoise
    w = polyfit(trainX, trainT, 9)
    fig = ax[idx]
    fig.set_title('M=%d' % val)
    fig.plot(xlab, ylab)
    fig.plot(trainX, trainT, 'ro')
    fig.plot(xlab, polynomial(w, xlab), 'g')


#Ridge regression

n_alphas = 200
start = -40
end = -20
lnAlpha = arange(start, end, float(end - start) / n_alphas)
alphas = exp(lnAlpha)
#for i in lnAlpha]//arange(start, end, float(end - start) / n_alphas)
#alphas = np.logspace(start, end, n_alphas, base = e)
clf = linear_model.Ridge(fit_intercept=True)

error = []
trainError = []

#myList = [([1. * i / N]) for i in range(N)]

powers = range(0, 9)

myList = array(x)[:,newaxis] ** array(powers)

for a in alphas:
    clf.set_params(alpha=a)
    model = clf.fit(myList, t)
    #y_plot = model.predict(xlab)
    weights = [model.intercept_] + model.coef_
    weightsInvserse = weights[::-1]
    error.append(rms(weightsInvserse, xTest, yTest))
    trainError.append(rms(weightsInvserse, x, t))

figAlphaError = plt.figure()
plt.xlabel("lambda")
plt.ylabel("RMS ERROR")
plt.plot(lnAlpha, error, label='test error')
plt.plot(lnAlpha, trainError, label='train error')

plt.show()
