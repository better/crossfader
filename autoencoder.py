import math
import random
import numpy
import json
import theano
import theano.tensor as T
from theano.tensor.nnet import sigmoid

def floatX(x):
    return numpy.asarray(x, dtype=theano.config.floatX)

data = json.load(open('stock-data.json'))

headers = sorted(list(set([key for v in data.values() for key in v.keys()])))
values = [[v[header] for v in data.values() if header in v] for header in headers]

D = len(data)
K = 1000 # Random splits
splits = []
for i in xrange(K):
    j = random.randint(0, len(headers)-1)
    splits.append((j, random.choice(values[j])))

def get_row(data_row):
    M_row = numpy.zeros(K)
    V_row = numpy.zeros(K)

    for k, split in enumerate(splits):
        j, x_split = split
        if headers[j] not in data_row:
            continue
        x = data_row[headers[j]]
        if x < x_split:
            M_row[k] = 1
        elif x > x_split:
            M_row[k] = 1
            V_row[k] = 1

    return M_row, V_row

M = numpy.zeros((D, K), dtype=theano.config.floatX)
V = numpy.zeros((D, K), dtype=theano.config.floatX)

for i, key in enumerate(data.keys()):
    M[i], V[i] = get_row(data[key])

def W_values(n_in, n_out):
    return numpy.random.uniform(
        low=-numpy.sqrt(6. / (n_in + n_out)),
        high=numpy.sqrt(6. / (n_in + n_out)),
        size=(n_in, n_out))

srng = theano.tensor.shared_randomstreams.RandomStreams()

def get_parameters():
    # Train an autoencoder to reconstruct the rows of the V matrices
    n_hidden_layers = 2
    n_hidden_units = 64
    Ws, bs = [], []
    for l in xrange(n_hidden_layers + 1):
        n_in, n_out = n_hidden_units, n_hidden_units
        if l == 0:
            n_in = K
        elif l == n_hidden_layers:
            n_out = K

        Ws.append(theano.shared(W_values(n_in, n_out)))
        gamma = 0.1 # initialize it to slightly positive so the derivative exists
        bs.append(theano.shared(numpy.ones(n_out) * gamma))

    return Ws, bs

def get_model(Ws, bs):
    m = T.matrix('mask')
    v = T.matrix('input')

    # Dropout in the input as well
    # use_input_dropout = T.scalar()
    m_dropped = m * srng.binomial(n=1, p=0.5, size=v.shape)
    h = v * m_dropped + 0.5 * (1 - m_dropped) # set all unknkown values to 0.5
    
    for l in xrange(len(Ws)):

        h = T.dot(h, Ws[l]) + bs[l]

        if l < len(Ws) - 1:
            h = h * (h > 0) # relu
            mask = srng.binomial(n=1, p=0.5, size=h.shape)
            h = h * mask * 2

    output = sigmoid(h)
    
    LL = v * T.log(output) + (1 - v) * T.log(1 - output)
    loss = -(m * LL).sum() / m.sum()

    return m, v, loss

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

Ws, bs = get_parameters()
m, v, loss = get_model(Ws, bs)
updates = nesterov_updates(loss, Ws + bs, 1e-1, 0.9)
loss_f = theano.function([m, v], loss, updates=updates)

for iter in xrange(1000000):
    print loss_f(M, V)

    if (iter + 1) % 20 == 0:
        W = Ws[0].get_value()

        def cos(a, b):
            p, q = W[a], W[b]
            return numpy.dot(p, q) / math.sqrt(numpy.dot(p, p) * numpy.dot(q, q))

        for a in xrange(5):
            bs = sorted(xrange(K), key=lambda b: cos(a, b), reverse=True)
            for b in bs[:10]:
                j, split_x = splits[b]
                print '%.4f %30s %15.2f' % (cos(a, b), headers[j], split_x)
            print

        # M_row, V_row = get_row(data['AAPL'])
        # print M_row, V_row
