import csv

# Using data from http://www.imf.org/external/pubs/ft/weo/2014/02/weodata/download.aspx

writer = csv

with open('examples/data/WEOOct2014all.csv', 'rU') as csvfile:
    reader = csv.reader(csvfile)
    headers = reader.next() # ignore headers

    series_by_country = {}
    for row in reader:
        country = row[3]
        metric = row[4]
        data = row[9:-1]

        if not metric:
            continue

        series_by_country.setdefault(country, {})[metric] = data

    data = []
    headers = set()
    for country, series in series_by_country.iteritems():
        # Go through all years now
        series_keys = list(series.keys())
        headers.update(series_keys)

        for values in zip(*[series[key] for key in series_keys]):
            data_row = {}
            for key, value in zip(series_keys, values):
                if value in ['', 'n/a', '--']:
                    continue

                data_row[key] = float(value.replace(',', ''))

            data.append(data_row)

headers = list(headers)

writer = csv.writer(open('examples/data/countries.csv', 'w'))
writer.writerow(headers)

for row in data:
    if not any([header in row for header in headers]):
        continue
    writer.writerow([row.get(header, '') for header in headers])
