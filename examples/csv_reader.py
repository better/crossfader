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
parser.add_argument('--delimiter', default=',',
                    help='Delimiter for input')
parser.add_argument('--start-col', type=int, default=0,
                    help='Column to start from')
parser.add_argument('--header-threshold', type=float, default=0.0,
                    help='Only include headers with at least this much data defined')

args = parser.parse_args()

with open(args.input, 'rb') as csvfile:
    r = csv.reader(csvfile, delimiter=args.delimiter)
    headers = r.next()[args.start_col:]
    data = []
    for row in r:
        data.append(row[args.start_col:])

# Figure out which headers are numerical/categorical
keep_header_js = {}
for j, header in enumerate(headers):
    num_values = []
    nnum_values = []
    for row in data:
        if j >= len(row) or row[j] == '':
            continue
        try:
            float(row[j])
            num_values.append(row[j])
        except ValueError:
            nnum_values.append(row[j])

    print header, len(num_values), len(nnum_values), len(set(num_values)), len(set(nnum_values))

    if len(num_values) + len(nnum_values) < len(data) * args.header_threshold:
        continue

    if len(set(num_values + nnum_values)) < args.bins:
        keep_header_js[j] = 'categorical'
    elif len(set(num_values)) > 1:
        keep_header_js[j] = 'numerical'
    
# Build matrix
keep_data = []
for row in data:
    keep_row = {}
    for j, t in keep_header_js.iteritems():
        h = headers[j]
        if j < len(row) and row[j] == '' and args.zero:
            keep_row[h] = 0.0
        elif j < len(row) and row[j] and t =='numerical':
            keep_row[h] = float(row[j])
        elif j < len(row) and row[j] and t =='categorical':
            keep_row[h] = row[j]

    if keep_row:
        keep_data.append(keep_row)

keep_headers = [headers[j] for j in keep_header_js.keys()]
keep_headers_types = dict([(headers[j], t) for j, t in keep_header_js.iteritems()])
print keep_headers_types

# Compress json output
json.encoder.FLOAT_REPR = lambda f: ("%.4f" % f)
json.encoder.c_make_encoder = None

print 'training'
for model in autoencoder.train(keep_headers, keep_data,
                               bins=args.bins,
                               n_hidden_units=args.n_hidden_units,
                               n_hidden_layers=args.n_hidden_layers,
                               headers_types=keep_headers_types):
    js = 'var data = %s;' % json.dumps(model)
    f = open(args.output, 'w')
    f.write(js)
    f.close()
