All these example datasets can be trained using the csv_reader.py

car
---

* `15tstcar-2014-11-13.csv`
* http://www.epa.gov/otaq/tcldata.htm
* Note the weird semicolon delimiter.

wine
----

* `winequality-white.csv`
* https://archive.ics.uci.edu/ml/datasets/Wine+Quality

countries
---------

* `WEOOct2014all.csv`
* http://www.imf.org/external/pubs/ft/weo/2014/01/weodata/download.aspx
* The source file is not the training data.
* You need to run the script `countries.py` to build the "countries.csv" dataset.

stocks
------

* `stocks.csv`
* This dataset is constructed from API queries to the Yahoo Finance API.
* See `stocks_fundamentals.py` for how to construct the dataset from scratch.
