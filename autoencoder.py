import math
import random
import numpy
import json
import theano
import theano.tensor as T
from theano.tensor.nnet import sigmoid

def floatX(x):
    return numpy.asarray(x, dtype=theano.config.floatX)

stock_data = json.load(open('stock-data.json'))

headers = set()
for k, v in stock_data.iteritems():
    headers.update(set(v.keys()))

headers = list(headers)

samples = [[] for h in headers]

for k, v in stock_data.iteritems():
    for j, m in enumerate(headers):
        if headers[j] in v:
            samples[j].append(v[headers[j]])

K = 1000 # Random splits
splits = []
for i in xrange(K):
    j = random.randint(0, len(headers)-1)
    x = random.choice(samples[j])
    splits.append((j, x))

M = numpy.zeros((len(stock_data), K), dtype=theano.config.floatX)
V = numpy.zeros((len(stock_data), K), dtype=theano.config.floatX)

for i, stock in enumerate(stock_data.keys()):
    for k, split in enumerate(splits):
        j, x = split
        if headers[j] not in stock_data[stock]:
            continue
        y = stock_data[stock][headers[j]]
        if y < x:
            M[i][k] = 1
            V[i][k] = 0
        elif y > x:
            M[i][k] = 1
            V[i][k] = 1


# Train an autoencoder to reconstruct the rows of the V matrices
n_hidden_layers = 4
n_hidden_units = 256

def W_values(n_in, n_out):
    return numpy.random.uniform(
        low=-numpy.sqrt(6. / (n_in + n_out)),
        high=numpy.sqrt(6. / (n_in + n_out)),
        size=(n_in, n_out))

srng = theano.tensor.shared_randomstreams.RandomStreams()

m = T.matrix('mask')
v = T.matrix('input')

# Dropout in the input as well
m_dropped = m * srng.binomial(n=1, p=0.5, size=v.shape)
h = v * m_dropped + 0.5 * (1 - m_dropped) # set all unknkown values to 0.5

params = []
for l in xrange(n_hidden_layers + 1):
    n_in, n_out = n_hidden_units, n_hidden_units
    if l == 0:
        n_in = K
    elif l == n_hidden_layers:
        n_out = K

    W_s = theano.shared(W_values(n_in, n_out))
    gamma = 0.1 # initialize it to slightly positive so the derivative exists
    b_s = theano.shared(numpy.ones(n_out) * gamma)

    params += [W_s, b_s]

    h = T.dot(h, W_s) + b_s
    #mask = srng.binomial(n=1, p=0.5, size=h.shape)
    #h = h * mask * 2

output = sigmoid(h)
    
LL = v * T.log(output) + (1 - v) * T.log(1 - output)
loss = -(m * LL).sum() / m.sum()

def nesterov_updates(loss, all_params, learn_rate, momentum):
    updates = []
    all_grads = T.grad(loss, all_params)
    for param_i, grad_i in zip(all_params, all_grads):
        # generate a momentum parameter
        mparam_i = theano.shared(numpy.array(param_i.get_value()*0.))
        v = momentum * mparam_i - learn_rate * grad_i
        w = param_i + momentum * v - learn_rate * grad_i
        updates.append((param_i, w))
        updates.append((mparam_i, v))
    return updates

updates = nesterov_updates(loss, params, 0.1, 0.9)
loss_f = theano.function([m, v], loss, updates=updates)

while True:
    print loss_f(M, V)
