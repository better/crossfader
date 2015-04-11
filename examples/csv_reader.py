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

# Figure out which headers are numerical
keep_header_js = []
for j, header in enumerate(headers):
    values = []
    for row in data:
        if row[j] == '':
            continue
        try:
            float(row[j])
        except ValueError:
            break
        values.append(row[j])

    if len(set(values)) > 1 and len(values) > len(data) * args.header_threshold:
        keep_header_js.append(j)
    
# Build matrix
keep_data = []
for row in data:
    keep_row = {}
    for j in keep_header_js:
        h = headers[j]
        if row[j] == '' and args.zero:
            keep_row[h] = 0.0
        elif row[j]:
            keep_row[h] = float(row[j])

    if keep_row:
        keep_data.append(keep_row)

keep_headers = [headers[j] for j in keep_header_js]

print 'training'
for model in autoencoder.train(keep_headers, keep_data,
                               bins=args.bins,
                               n_hidden_units=args.n_hidden_units,
                               n_hidden_layers=args.n_hidden_layers):
    js = 'var data = %s;' % json.dumps(model)
    f = open(args.output, 'w')
    f.write(js)
    f.close()
