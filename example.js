var RandomForest = require('./random-forest'),
    fs = require('fs');

function readExample(callback) {
  var rawData = require('./stock-data.json');
  var headers = {};
  for (k in rawData)
    for (m in rawData[k])
      headers[m] = true;

  headers = Object.keys(headers);
  
  var data = Object.keys(rawData).map(function(k) {
	  return headers.map(function(m) { return rawData[k][m]; });
      });

  return {'headers': headers, 'data': data};
}

function crossValidate(data, nEstimators, useMedian) {
  var nSamples = 99;

  if (useMedian) {
    var sumDiff = [];
    var nObs = [];

    for (var j = 0; j < data[0].length; j++) {
	sumDiff.push(0.0);
	nObs.push(0.0);
    }

  } else {
    var distribution = [];

    for (var j = 0; j < (nSamples + 1); j++)
      distribution.push(j);
  }

  // Fit model to dump
  var model = new RandomForest();
  model.train(data, nEstimators, useMedian, 100, 10);
  var serialized = JSON.stringify(model.serialize(), function(key, val) {
    if (val && val.toPrecision)  // reduce file size by stripping digits
      return val.toPrecision(3);
    else
      return val;
  });
  serialized = 'var data = ' + serialized + ';';
  fs.writeFile('model-' + nEstimators + '-' + useMedian + '.js', serialized);

  // Do cross validation
  for (var kFold = 0; kFold < 10; kFold++) {

    var trainData = [];
    var testData = [];

    for (var i = 0; i < data.length; i++){
      if (Math.random() < 0.5)
	trainData.push(data[i]);
      else
	testData.push(data[i]);
    }

    // Fit model
    var model = new RandomForest();
    model.train(trainData, nEstimators, useMedian, 100, 10);

    // Predict missing values from the test set
    for (var i = 0; i < testData.length; i++) {
      for (var j = 0; j < testData[i].length; j++) {
	if (testData[i][j] == undefined)
	  continue;

	if (useMedian)
	  nObs[j]++;

	var samples = [];
	for (var sample = 0; sample < nSamples; sample++) {
	  var row = testData[i].slice(0);
	  row[j] = undefined;
	  samples.push(model.fill(row)[j]);
	}
	samples.sort(function(a, b) { return a-b; });

	if (useMedian) {
	  sumDiff[j] += Math.abs(testData[i][j] - samples[Math.floor(samples.length/2)]);
	} else {
	  // Find the bucket of the real value
	  var bucketLo = -1, bucketHi = nSamples;
	  for (var b = 0; b < nSamples; b++) {
	    if (samples[b] < testData[i][j])
	      bucketLo = b;
	    if (samples[b] > testData[i][j] && bucketHi == nSamples)
	      bucketHi = b;
	  }

	  for (var b = bucketLo+1; b < bucketHi+1; b++)
	    distribution[b] += 1.0 / (bucketHi - bucketLo);

	  // console.log(test[i][j] + ' ' + samples + ' ' + bucketLo + '-' + bucketHi);
	}
      }
    }
  }

  console.log(nEstimators + ' ' + (useMedian ? 'median' : 'distribution'));
  if (useMedian) {
    for (var j = 0; j < sumDiff.length; j++) {
      console.log(j + ': ' + (sumDiff[j] / nObs[j]).toFixed(2));
    }
  } else {
    console.log(distribution);
  }

  console.log();
}

var stuff = readExample();
fs.writeFile('headers.js', 'var headers = ' + JSON.stringify(stuff['headers']) + ';');
/*
var nEstimators = [1, 10, 100, 1000];
for (var useMedian = 0; useMedian < 2; useMedian++) {
  for (var i = 0; i < nEstimators.length; i++) {
    crossValidate(stuff['data'], nEstimators[i], useMedian == 1);
  }
}

*/
