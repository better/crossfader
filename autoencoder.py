#  Copyright 2015 One Zero Capital
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import math
import random
import numpy
import json
import theano
import theano.tensor as T
from theano.tensor.nnet import sigmoid
import time
import bisect

def floatX(x):
    return numpy.asarray(x, dtype=theano.config.floatX)

srng = theano.tensor.shared_randomstreams.RandomStreams()

def get_splits(headers, data, bins, linear=False):
    values = [[v[header] for v in data if header in v] for header in headers]

    splits = []
    for j in xrange(len(headers)):
        values_j_unique = sorted(set(values[j]))
        lo, hi = numpy.percentile(values_j_unique, 1.0), numpy.percentile(values_j_unique, 99.0)
        print '%100s %11.2f %11.2f %5.2f%%' % (headers[j], lo, hi, 100. * len(values[j]) / len(data))
        j_splits = []
        if linear:
            for bin in xrange(bins):
                j_splits.append(lo + (bin + 1) * (hi - lo) * 1. / bins)
        else:
            for bin in xrange(bins):
                j_splits.append(numpy.percentile(values_j_unique, 100.0 * (bin+1) / (bins+1)))

        splits += [(j, x_split) for x_split in j_splits]
            
    return splits

        
def get_row(headers, K, data_row, splits, headers_keep=None):
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

        if headers_keep is not None:
            if headers[j] not in headers_keep:
                Q_row[k] = 1

    return V_row, M_row, Q_row


def build_matrices(headers, data, D, K, splits, batch_size=200):
    V = numpy.zeros((D, K), dtype=theano.config.floatX)
    M = numpy.zeros((D, K), dtype=theano.config.floatX)
    Q = numpy.zeros((D, K), dtype=theano.config.floatX)
    k = numpy.zeros((D, ), dtype=theano.config.floatX)

    for i, data_row in enumerate(random.sample(data, batch_size)):
        # How many header should we remove
        n_headers_keep = random.randint(0, len(headers))
        headers_keep = set(random.sample(headers, n_headers_keep))
        V[i], M[i], Q[i] = get_row(headers, K, data_row, splits, headers_keep)
        k[i] = len([h for h in headers if h in headers_keep and data_row])

    return V, M, Q, k


def W_values(n_in, n_out):
    return numpy.random.uniform(
        low=-numpy.sqrt(6. / (n_in + n_out)),
        high=numpy.sqrt(6. / (n_in + n_out)),
        size=(n_in, n_out))


def get_parameters(K, n_hidden_layers=4, n_hidden_units=128):
    # Train an autoencoder to reconstruct the rows of the V matrices
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
    k = T.vector('normalization factor')

    # Set all missing/target values to 0.5
    keep_mask = (1-m) * (1-q)
    h = keep_mask * (v * 2 - 1) # Convert to +1, -1
    
    # Normalize layer 0
    h *= (k.dimshuffle(0, 'x') + 1)** -0.5

    for l in xrange(len(Ws)):
        h = T.dot(h, Ws[l]) + bs[l]

        if l < len(Ws) - 1:
            h = h * (h > 0) # relu
            if dropout:
                mask = srng.binomial(n=1, p=0.5, size=h.shape)
                h = h * mask * 2

    output = sigmoid(h)
    LL = v * T.log(output) + (1 - v) * T.log(1 - output)
    # loss = -(q * LL).sum() / q.sum()
    loss = -((1 - m) * LL).sum() / (1 - m).sum()

    return v, m, q, k, output, loss


def nesterov_updates(loss, all_params, learn_rate, momentum, weight_decay):
    updates = []
    all_grads = T.grad(loss, all_params)
    for param_i, grad_i in zip(all_params, all_grads):
        # generate a momentum parameter
        mparam_i = theano.shared(numpy.array(param_i.get_value()*0.))
        full_grad_i = grad_i + learn_rate * weight_decay * param_i
        v = momentum * mparam_i - learn_rate * full_grad_i
        w = param_i + momentum * v - learn_rate * full_grad_i
        updates.append((param_i, w))
        updates.append((mparam_i, v))
    return updates


def get_train_f(Ws, bs):
    learning_rate = T.scalar('learning rate')
    v, m, q, k, output, loss = get_model(Ws, bs, dropout=False)
    updates = nesterov_updates(loss, Ws + bs, learning_rate, 0.9, 1e-6)
    return theano.function([v, m, q, k, learning_rate], loss, updates=updates)


def get_pred_f(Ws, bs):
    v, m, q, k, output, loss = get_model(Ws, bs, dropout=False)
    return theano.function([v, m, q, k], output)


def train(headers, data, n_hidden_layers=4, n_hidden_units=128, bins=40):
    D = len(data)
    K = bins * len(headers)

    print D, 'data points', K, 'random splits', bins, 'bins', K, 'features'

    splits = get_splits(headers, data, bins)

    Ws, bs = get_parameters(K, n_hidden_layers, n_hidden_units)
    train_f = get_train_f(Ws, bs)
    pred_f = get_pred_f(Ws, bs)

    learning_rate = 1.0
    n_iters_patience = 1000
    avg_decay = 1.0 - 1.0 / n_iters_patience
    loss_sum = 0.0
    weight_sum = 0.0
    best_loss_smoothed = float('inf')
    best_iter = 0
    
    for iter in xrange(1000000):
        V, M, Q, k = build_matrices(headers, data, D, K, splits)
        loss = train_f(V, M, Q, k, learning_rate)
        loss_sum = loss_sum * avg_decay + loss
        weight_sum = weight_sum * avg_decay + 1.0

        loss_smoothed = loss_sum / weight_sum
        print '%12.9f %12.9f %5d %5d %12.9f' % (loss_smoothed, loss, iter, iter-best_iter, learning_rate)

        if loss_smoothed < best_loss_smoothed:
            best_iter = iter
            best_loss_smoothed = loss_smoothed

        if iter > best_iter + n_iters_patience:
            print 'lower learning rate'
            learning_rate *= 0.3
            best_loss_smoothed = float('inf')
            if learning_rate < 1e-4:
                break

        if (iter + 1) % 10 == 0:
            yield {'K': K, 'bins': bins, 'splits': splits, 'headers': headers,
                   'Ws': [W.get_value().tolist() for W in Ws],
                   'bs': [b.get_value().tolist() for b in bs]}


if __name__ == '__main__':
    data = json.load(open('stock-data.json'))

    headers = sorted(list(set([key for v in data.values() for key in v.keys()])))

    train(headers, data)

