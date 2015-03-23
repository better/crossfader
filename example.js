var RandomForest = require('./random_forests'),
    fs = require('fs');

function readExample(callback) {
  var rawData = require('./data.json');
  var headers = {};
  for (k in rawData)
    for (m in rawData[k])
      headers[m] = true;

  var data = Object.keys(rawData).map(function(k) {
	  return Object.keys(headers).map(function(m) { return rawData[k][m]; });
      });

  return data;
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


  for (var kFold = 0; kFold < 10; kFold++) {
    // Do cross validation
    var trainData = [];
    var testData = [];

    for (var i = 0; i < data.length; i++){
      if (Math.random() < 0.5)
	trainData.push(data[i]);
      else
	testData.push(data[i]);
    }

    // Fit model
    var model = new RandomForest(nEstimators, useMedian);
    model.train(trainData);

    // Predict missing values from the test set
    for (var i = 0; i < testData.length; i++) {
      for (var j = 0; j < testData[i].length; j++) {
	if (testData[i][j] == null)
	  continue;

	if (useMedian)
	  nObs[j]++;

	var samples = [];
	for (var sample = 0; sample < nSamples; sample++) {
	  var row = testData[i].slice(0);
	  row[j] = null;
	  samples.push(model.fill(row)[j]);
	}
	samples.sort(function(a, b) { return a-b;} );

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

  console.log(nEstimators + ' ' + useMedian);
  if (useMedian) {
    for (var j = 0; j < sumDiff.length; j++) {
      console.log(j + ': ' + (sumDiff[j] / nObs[j]).toFixed(2));
    }
  } else {
    console.log(distribution);
  }

  console.log();
}

var data = readExample();
var nEstimators = [1, 10, 100, 1000];
for (var useMedian = 0; useMedian < 2; useMedian++) {
  for (var i = 0; i < nEstimators.length; i++) {
    console.log(nEstimators[i] + ' ' + useMedian ? 'median' : 'distribution');
    crossValidate(data, nEstimators[i], useMedian == 1);
  }
}

