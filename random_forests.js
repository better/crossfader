var readline = require('readline');

// Hardcoded shit for now
var nEstimators = 10;
var maxFeatures = 100; // Sampling from k^2 feature pairs
var minLeaf = 10;

function readStdin(callback) {
  var data = [];

  var rl = readline.createInterface({input: process.stdin});

  rl.on('line', function(line){
    var y = [];
    line.split(' ').map(function(item) {
      if (item)
	y.push(parseFloat(item))
    })
    data.push(y);
  })
  
  rl.on('close', function() {
    callback(data);
  })
}

function bootstrap(data) {
  var dataSampled = [];
  for (var i = 0; i < data.length;i++)
    dataSampled.push(data[Math.floor(Math.random() * data.length)]);
  return dataSampled;
}

function fitTree(data) {
  // Returns a classifier
  if (data.length <= minLeaf)
    return function(row) { return data[Math.floor(Math.random() * data.length)]; }

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
    var c = [[1, 1], [1, 1]]; // Initializing with 1 for reasons beyond this comment. Has to do with Beta distributions.
    for (var j = 0; j < data.length; j++)
      c[data[j][fx] < bx ? 1 : 0][data[j][fy] < by ? 1 : 0]++;

    // Calculate information gain
    function entropy(p) { return -p * Math.log(p) - (1 - p) * Math.log(1-p); }
    var H1 = entropy((c[0][0] + c[1][0]) / data.length);
    var H2 = (c[0][0] + c[0][1]) / data.length * (entropy(c[0][0] / (c[0][0] + c[0][1]))) + (c[1][0] + c[1][1]) / data.length * (entropy(c[1][0] / (c[1][0] + c[1][1])));

    var ig = H1 - H2;

    if (!isNaN(ig) && ig > bestIg) {
      bestIg = ig;
      function createFunc(fx, bx) {
	var _fx = fx, _bx = bx;
	return function(row) { return row[_fx] < _bx ? 1 : 0; };
      }
      bestFunc = createFunc(fx, bx);
    }
  }

  if (bestIg == 0) {
    return function(row) { return data[Math.floor(Math.random() * data.length)]; }
  }

  var dataSplit = [[], []];
  for (var i = 0; i < data.length; i++)
    dataSplit[bestFunc(data[i])].push(data[i]);

  var funcs = dataSplit.map(fitTree);

  return function(row) { return funcs[bestFunc(row)](row); }
}

function fit(data) {
  estimators = [];
  for (var i = 0; i < nEstimators; i++)
    estimators.push(fitTree(data));

  
  tree = fitTree(data);
  console.log(data[0]);
  console.log(tree(data[0]));
}

readStdin(fit);
