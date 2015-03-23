import ystockquote
import json

symbols = json.load(open('symbols.json'))
data = {}

for symbol in symbols:
    print symbol, '...'
    data[symbol] = {}
    for k, v in ystockquote.get_all(symbol).iteritems():
        if v[-1] == 'B':
            v, f = v[:-1], 1e9
        elif v[-1] == 'M':
            v, f = v[:-1], 1e6
        else:
            f = 1
        try:
            data[symbol][k] = float(v) * f
        except ValueError:
            print 'could not parse', k, v

json.dump(data, open('stock-data.json', 'w'))
        
