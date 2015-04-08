import csv
import sys
import argparse
import autoencoder
import json

parser = argparse.ArgumentParser()
parser.add_argument('--input',
                    help='File to read from')
parser.add_argument('--zero', action='store_true',
                    help='Store empty strings as zeros')
parser.add_argument('--output',
                    help='File to dump model to')
parser.add_argument('--bins', type=int, default=20,
                    help='Number of bins for the histogram')
parser.add_argument('--n-hidden-layers', type=int, default=3,
                    help='Number of hidden layers')
parser.add_argument('--n-hidden-units', type=int, default=64,
                    help='Number of hidden units in each layer')

args = parser.parse_args()

with open(args.input, 'rb') as csvfile:
    r = csv.reader(csvfile)
    headers = r.next()
    data = []
    for row in r:
        data.append(row)

# Figure out which headers are numerical
num_header_js = []
for j, header in enumerate(headers):
    unique = set()
    for row in data:
        if args.zero and row[j] == '':
            continue
        try:
            float(row[j])
        except ValueError:
            break
        unique.add(row[j])

    else:
        if len(unique) > 1:
            num_header_js.append(j)
    
# Build matrix
num_data = []
for row in data:
    num_row = {}
    for j in num_header_js:
        h = headers[j]
        if row[j] == '' and args.zero:
            num_row[h] = 0.0
        elif row[j]:
            num_row[h] = float(row[j])

    if num_row:
        num_data.append(num_row)

num_headers = [headers[j] for j in num_header_js]

print 'training'
for model in autoencoder.train(num_headers, num_data,
                               bins=args.bins,
                               n_hidden_units=args.n_hidden_units,
                               n_hidden_layers=args.n_hidden_layers):
    js = 'var data = %s;' % json.dumps(model)
    f = open(args.output, 'w')
    f.write(js)
    f.close()
