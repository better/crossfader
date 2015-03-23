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
    var r = undefined;
    for (var i = 0; i < data.length; i++)
      if (data[i][j] !== undefined)
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
      if (data[i][j] !== undefined)
	elms.push(data[i][j]);

    if (elms.length % 2)
      result.push(elms[Math.floor(elms.length/2)]);
    else
      result.push(elms[Math.floor((elms.length + Math.random())/2)]);
  }
  return result;
}

function copyDefined(src, dst) {
  for (var j = 0; j < dst.length; j++)
    if (dst[j] === undefined)
      dst[j] = src[j];
  return dst;
}

function SplitPoint(fx, bx, pMissing) {
  this._fx = fx;
  this._bx = bx;
  this._pTies = Math.random();
  this._pMissing = pMissing;
}

SplitPoint.prototype.getSide = function(row) {
  if (row[this._fx] == undefined)
    return (Math.random() < this._pMissing) ? 1 : 0;
  else if (row[this._fx] == this._bx)
    return (Math.random() < this._pTies) ? 1 : 0;
  else
    return (row[this._fx] < this._bx) ? 1 : 0;
}

SplitPoint.prototype.serialize = function() {
  return {'bx': this._bx, 'fx': this._fx, 'pm': this._pMissing, 'pt': this._pTies};
}

function Leaf(representativeRow) {
  this._representativeRow = representativeRow;
}

Leaf.prototype.fill = function(row) {
 return copyDefined(this._representativeRow, row.slice(0));
}

Leaf.prototype.serialize = function() {
  return {'type': 'leaf', 'row': this._representativeRow};
}

function NonLeaf(split, lChild, rChild) {
  this._split = split;
  this._lChild = lChild;
  this._rChild = rChild;
}

NonLeaf.prototype.fill = function(row) {
  return [this._lChild, this._rChild][this._split.getSide(row)].fill(row);
}

NonLeaf.prototype.serialize = function() {
  return {'type': 'nonleaf', 'split': this._split.serialize(), 'L': this._lChild.serialize(), 'R': this._rChild.serialize()};
}

function RandomForest(nEstimators, useMedian) {
  this._nEstimators = nEstimators;
  this._useMedian = useMedian;
  this._estimators = [];

  // Hardcoded shit for now
  this._maxFeatures = 100; // Sampling from k^2 feature pairs
  this._minLeaf = 10;
}

RandomForest.prototype.fitTree = function(data, useMedian) {
  // Returns a classifier

  // Compute an representative feature vector
  if (useMedian)
    var representativeRow = median(data);
  else
    var representativeRow = sampleRow(data);

  // Handle leaf nodes (too few data points)
  if (data.length <= this._minLeaf) {
    return new Leaf(representativeRow);
  }

  var nFeatures = data[0].length;
  var bestIg = 0.0, bestSplit = undefined;
  for (var i = 0; i < this._maxFeatures; i++) {
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
      bestSplit = new SplitPoint(fx, bx, pMissing);
    }
  }

  if (bestIg == 0)
    return new Leaf(representativeRow);

  var dataSplit = [[], []];
  for (var i = 0; i < data.length; i++)
    dataSplit[bestSplit.getSide(data[i])].push(data[i]);

  if (dataSplit[0].length == 0 || dataSplit[1].length == 0)
    return new Leaf(representativeRow);

  var children = dataSplit.map(this.fitTree, useMedian);

  return new NonLeaf(bestSplit, children[0], children[1]);
}

RandomForest.prototype.train = function(data) {
  for (var i = 0; i < this._nEstimators; i++) {
    var bootstrappedData = bootstrap(data);
    this._estimators.push(this.fitTree(bootstrappedData, this._useMedian));
  }
}

RandomForest.prototype.fill = function(row) {
  return this._estimators[Math.floor(Math.random() * this._nEstimators)].fill(row);
}

RandomForest.prototype.serialize = function() {
  return {'estimators': this._estimators.map(function(e) { return e.serialize(); })};
}

module.exports = RandomForest;
