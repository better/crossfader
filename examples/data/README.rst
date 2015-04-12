All these example datasets can be trained using the csv_reader.py

car
---

* 15tstcar-2014-11-13.csv
* http://www.epa.gov/otaq/tcldata.htm
* Note the weird semicolon delimiter.

wine
----

* winequality-white.csv
* https://archive.ics.uci.edu/ml/datasets/Wine+Quality

food
----

* sr26_food_db.csv
* http://www.ars.usda.gov/Services/docs.htm?docid=23634

countries
---------

* WEOOct2014all.csv
* http://www.imf.org/external/pubs/ft/weo/2014/01/weodata/download.aspx
* The source file is not the training data.
* You need to run the script "countries.py" to build the "countries.csv" dataset.

stocks
------

* stocks.csv
* This is not a raw dataset, but is constructed from API queries to the Yahoo Finance API.
* See stocks.py for how to construct the dataset from scratch.
