var random_forests = require('./random_forests'),
    readline = require('readline');

function readStdin(callback) {
  var data = [];

  var rl = readline.createInterface({input: process.stdin});

  rl.on('line', function(line){
    var y = [];
    line.split(' ').map(function(item) {
      if (item)
	y.push(parseFloat(item));
    })
    data.push(y);
  })
  
  rl.on('close', function() {
    callback(data);
  })
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
    var train = [];
    var test = [];

    for (var i = 0; i < data.length; i++){
      if (Math.random() < 0.5)
	train.push(data[i]);
      else
	test.push(data[i]);
    }

    // Fit model
    var model = random_forests(train, nEstimators, useMedian);

    // Predict missing values from the test set
    for (var i = 0; i < test.length; i++) {
      for (var j = 0; j < test[i].length; j++) {
	if (test[i][j] == null)
	  continue;

	if (useMedian)
	  nObs[j]++;

	var samples = [];
	for (var sample = 0; sample < nSamples; sample++) {
	  var row = test[i].slice(0);
	  row[j] = null;
	  samples.push(model(row)[j]);
	}
	samples.sort(function(a, b) { return a-b;} );

	if (useMedian) {
	  sumDiff[j] += Math.abs(test[i][j] - samples[Math.floor(samples.length/2)]);
	} else {
	  // Find the bucket of the real value
	  var bucketLo = -1, bucketHi = nSamples;
	  for (var b = 0; b < nSamples; b++) {
	    if (samples[b] < test[i][j])
	      bucketLo = b;
	    if (samples[b] > test[i][j] && bucketHi == nSamples)
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

readStdin(function(data) {
  var nEstimators = [1, 10, 100, 1000];
  for (var useMedian = 0; useMedian < 2; useMedian++) {
    for (var i = 0; i < nEstimators.length; i++) {
      crossValidate(data, nEstimators[i], useMedian == 1);
    }
  }
});
