// Hardcoded shit for now
var maxFeatures = 100; // Sampling from k^2 feature pairs
var minLeaf = 10;

function bootstrap(data) {
  var dataSampled = [];
  for (var i = 0; i < data.length;i++)
    dataSampled.push(data[Math.floor(Math.random() * data.length)]);
  return dataSampled;
}

function sampleRow(data) {
  // Sample non-null elements from data
  var result = [];
  for (var j = 0; j < data[0].length; j++) {
    var k = 1;
    var r = null;
    for (var i = 0; i < data.length; i++)
      if (data[i][j] !== null)
	if (Math.random() * k++ < 1.0)
	  r = data[i][j];
    result.push(r);
  }
  return result;
}

function median(data) {
  // Sample non-null elements from data
  var result = [];
  for (var j = 0; j < data[0].length; j++) {
    var elms = []
    for (var i = 0; i < data.length; i++)
      if (data[i][j] !== null)
	elms.push(data[i][j]);

    if (elms.length % 2)
      result.push(elms[Math.floor(elms.length/2)]);
    else
      result.push(elms[Math.floor((elms.length + Math.random())/2)]);
  }
  return result;
}

function copyNonNull(src, dst) {
  for (var j = 0; j < dst.length; j++)
    if (dst[j] === null)
      dst[j] = src[j];
  return dst;
}

function fitTree(data, useMedian) {
  // Returns a classifier

  // Compute an representative feature vector
  if (useMedian)
    var representativeRow = median(data);
  else
    var representativeRow = sampleRow(data);

  // Handle leaf nodes (too few data points)
  if (data.length <= minLeaf) {
    return function(row) { return copyNonNull(representativeRow, row.slice(0)); }
  }

  var nFeatures = data[0].length;
  var bestIg = 0.0, bestFunc = null;
  for (var i = 0; i < maxFeatures; i++) {
    // Sample features
    var fx = Math.floor(Math.random() * nFeatures);
    var fy = Math.floor(Math.random() * nFeatures);

    // Sample split points
    var bx = data[Math.floor(Math.random() * data.length)][fx];
    var by = data[Math.floor(Math.random() * data.length)][fy];

    // Count statistics
    var c = [[0.0, 0.0], [0.0, 0.0]];
    for (var j = 0; j < data.length; j++)
      c[data[j][fx] < bx ? 1 : 0][data[j][fy] < by ? 1 : 0]++;

    // Calculate information gain
    function entropy(p) { return -p * Math.log(p) - (1 - p) * Math.log(1-p); }
    var H1 = entropy((c[0][0] + c[1][0]) / data.length);
    var H2 = (c[0][0] + c[0][1]) / data.length * (entropy(c[0][0] / (c[0][0] + c[0][1]))) + (c[1][0] + c[1][1]) / data.length * (entropy(c[1][0] / (c[1][0] + c[1][1])));

    var ig = H1 - H2;
    var pMissing = (c[0][0] + c[0][1] + 1.0) / (data.length + 2.0); // Mean of posterior assuming a Beta(1, 1) prior
    var pTies = Math.random(); // This ensures the split is uniform in the presense of points with infinite probability distribution

    if (!isNaN(ig) && ig > bestIg) {
      bestIg = ig;
      function createFunc(fx, bx, pTies, pMissing) {
	var _fx = fx, _bx = bx, _pTies = pTies, _pMissing = pMissing;
	return function(row) {
	  if (row[_fx] == null)
	    return (Math.random() < _pMissing) ? 1 : 0;
	  else if (row[_fx] == _bx)
	    return (Math.random() < _pTies) ? 1 : 0;
	  else
	    return (row[_fx] < _bx) ? 1 : 0;
	};
      }
      bestFunc = createFunc(fx, bx, pTies, pMissing);
    }
  }

  if (bestIg == 0) {
    return function(row) { return copyNonNull(representativeRow, row.slice(0)); }
  }

  var dataSplit = [[], []];
  for (var i = 0; i < data.length; i++)
    dataSplit[bestFunc(data[i])].push(data[i]);

  if (dataSplit[0].length == 0 || dataSplit[1].length == 0) {
    return function(row) { return copyNonNull(representativeRow, row.slice(0)); }
  }

  var funcs = dataSplit.map(fitTree, useMedian);

  return function(row) {
    var result = funcs[bestFunc(row)](row);
    return copyNonNull(representativeRow, result);
  }
}

function fit(data, nEstimators, useMedian) {
  estimators = [];
  for (var i = 0; i < nEstimators; i++) {
    var bootstrappedData = bootstrap(data);
    estimators.push(fitTree(bootstrappedData, useMedian));
  }

  function sample(row, nSamples) {
    return estimators[Math.floor(Math.random() * estimators.length)](row);
  }

  return sample;
}

module.exports = fit;
