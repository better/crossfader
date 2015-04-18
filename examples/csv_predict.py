import csv
import sys
import argparse
import autoencoder
import json

parser = argparse.ArgumentParser()
parser.add_argument('--input',
                    help='File to read from')
parser.add_argument('--model',
                    help='Model json')
parser.add_argument('--output',
                    help='File to dump predicted data')
parser.add_argument('--write-cols', type=int, action='append',
                    help='Cols to write')

args = parser.parse_args()

line = open(args.model).next()
model = json.loads(line[line.find('{') : (line.rfind('}') + 1)])

data = []
with open(args.input, 'rb') as csvfile:
    r = csv.reader(csvfile) # , delimiter=args.delimiter)
    headers = r.next()
    for row in r:
        values = dict(zip(headers, row))
        for h in model['headers']:
            if h in values:
                if model['headers_types'].get(h) != 'categorical':
                    values[h] = float(values[h])

        data.append(values)

pred_f = autoencoder.get_pred_f(model['Ws'], model['bs'])
V, M, Q, k = autoencoder.build_matrices(model['headers'], data, model['K'], model['splits'], batch_size=999999999999)
values = pred_f(V, M, Q, k)

writer = csv.writer(open(args.output, 'w'))

all_headers = headers # will also include dropped headers
for header in model['headers']:
    if header not in all_headers:
        all_headers.append(header)

for row, v in zip(data, values):
    cdfs = [[] for h in model['headers']]
    for i, jx in enumerate(model['splits']):
        j, sign, x = jx
        # TODO: ONLY HANDLES NUMERICAL ATTRIBUTES
        cdfs[j].append((x, v[i]))

    for j, header in enumerate(model['headers']):
        if header not in row:
            if model['headers_types'].get(header) == 'categorical':
                probs = dict(cdfs[j])
                row[header] = ','.join([str(probs[cat]) for cat in probs])
            else:
                avg = 0
                cdfs_j = [(cdfs[j][0][0], 0.0)] + cdfs[j] + [(cdfs[j][-1][0], 1.0)]
                for i in xrange(len(cdfs_j) - 1):
                    xa, ca = cdfs_j[i]
                    xb, cb = cdfs_j[i+1]
                    avg += (cb - ca) * (xa + xb) / 2
                row[header] = avg

    row = [row.get(header, '') for header in all_headers]
    row = [row[col] for col in args.write_cols]
    writer.writerow(row)


    # print row, v
    
