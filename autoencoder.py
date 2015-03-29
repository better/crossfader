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

def get_row(data_row, headers_keep=None):
    # V: values
    V_row = numpy.zeros(K, dtype=theano.config.floatX)
    # M: what values are missing
    M_row = numpy.zeros(K, dtype=theano.config.floatX)
    # Q: what values to predict
    Q_row = numpy.zeros(K, dtype=theano.config.floatX)

    for k, split in enumerate(splits):
        j, x_split = split
        if headers[j] not in data_row:
            M_row[k] = 1
            continue
        x = data_row[headers[j]]
        if x < x_split:
            V_row[k] = 1

        if headers_keep is not None and headers[j] not in headers_keep:
            Q_row[k] = 1

    return V_row, M_row, Q_row

def build_matrices():
    V = numpy.zeros((D, K), dtype=theano.config.floatX)
    M = numpy.zeros((D, K), dtype=theano.config.floatX)
    Q = numpy.zeros((D, K), dtype=theano.config.floatX)

    for i, key in enumerate(data.keys()):
        # How many header should we remove
        n_headers_keep = random.randint(0, len(headers))
        headers_keep = set(random.sample(headers, n_headers_keep))
        V[i], M[i], Q[i] = get_row(data[key], headers_keep)

    return V, M, Q

def W_values(n_in, n_out):
    return numpy.random.uniform(
        low=-numpy.sqrt(6. / (n_in + n_out)),
        high=numpy.sqrt(6. / (n_in + n_out)),
        size=(n_in, n_out))

srng = theano.tensor.shared_randomstreams.RandomStreams()

def get_parameters():
    # Train an autoencoder to reconstruct the rows of the V matrices
    n_hidden_layers = 3
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

def get_model(Ws, bs, dropout=False):
    v = T.matrix('input')
    m = T.matrix('missing')
    q = T.matrix('target')

    # Set all missing/target values to 0.5
    keep_mask = (1-m) * (1-q)
    h = v * keep_mask + 0.5 * (1 - keep_mask)
    
    for l in xrange(len(Ws)):
        h = T.dot(h, Ws[l]) + bs[l]

        if l < len(Ws) - 1:
            h = h * (h > 0) # relu
            if dropout:
                mask = srng.binomial(n=1, p=0.5, size=h.shape)
                h = h * mask * 2

    output = sigmoid(h)
    
    LL = v * T.log(output) + (1 - v) * T.log(1 - output)
    loss = -(q * LL).sum() / q.sum()

    return v, m, q, output, loss

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

def get_train_f(Ws, bs):
    v, m, q, output, loss = get_model(Ws, bs, dropout=True)
    updates = nesterov_updates(loss, Ws + bs, 1e-1, 0.9)
    return theano.function([v, m, q], loss, updates=updates)

def get_pred_f(Ws, bs):
    v, m, q, output, loss = get_model(Ws, bs, dropout=False)
    return theano.function([v, m, q], output)

Ws, bs = get_parameters()
train_f = get_train_f(Ws, bs)
pred_f = get_pred_f(Ws, bs)

for iter in xrange(1000000):
    V, M, Q = build_matrices()
    print train_f(V, M, Q)

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

        V_row, M_row, Q_row = [x.reshape((1, K)) for x in get_row(data['AAPL'])]
        V_row_recon = pred_f(V_row, M_row, Q_row).astype(theano.config.floatX)

        print V_row[0,:10]
        print V_row_recon[0,:10]
        
        cdfs = [[] for h in headers]
        for i, split in enumerate(splits):
            j, x_split = split
            cdfs[j].append((x_split, V_row_recon[0][i]))

        for j, header in enumerate(headers):
            print header, sorted(cdfs[j])
