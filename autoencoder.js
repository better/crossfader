/*
  Copyright 2015 One Zero Capital

  Licensed under the Apache License, Version 2.0 (the "License");
  you may not use this file except in compliance with the License.
  You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

  Unless required by applicable law or agreed to in writing, software
  distributed under the License is distributed on an "AS IS" BASIS,
  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
  See the License for the specific language governing permissions and
  limitations under the License.
*/

Autoencoder = function(headers, splits, Ws, bs) {
  this.headers = headers;
  this.splits = splits;
  this.Ws = Ws;
  this.bs = bs;
  this.splitsByJ = [];
  for (var i = 0; i < this.splits.length; i++) {
    var j = this.splits[i][0];
    var x = this.splits[i][1];
    while (this.splitsByJ.length <= j) this.splitsByJ.push([]);
    this.splitsByJ[j].push(x);
  }

  this.Ds = []; // Smoothing factors
  for (var j = 0; j < this.splitsByJ.length; j++) {
    this.Ds.push([]);
    for (var i = 0; i < this.splitsByJ[j].length; i++) {
      var iLo = i > 0 ? i - 1 : i;
      var iHi = i < this.splitsByJ[j].length - 1 ? i + 1 : i;
      this.Ds[j].push((this.splitsByJ[j][iHi] - this.splitsByJ[j][iLo]) / (iHi - iLo));
    }
  }
}

Autoencoder.deserialize = function(data) {
  return new Autoencoder(data['headers'],
			 data['splits'],
			 data['Ws'],
			 data['bs']);
}

Autoencoder.prototype.getOutput = function(row) {
  var input = [];
  for (var j = 0; j < this.splitsByJ.length; j++) {
    for (var i = 0; i < this.splitsByJ[j].length; i++) {
      var x = this.splitsByJ[j][i];
      // Instead of encoding it as a -1 or +1 input, we smooth it using a tanh representation
      // See description in getPdfs, it's the same idea
      var D = this.Ds[j][i];

      if (row[j] == undefined) {
	input.push(0.0);
      } else {
	input.push(Math.tanh((x - row[j]) / D));
      }
    }
  }

  // Normalize input
  var k = 0;
  for (var j = 0; j < row.length; j++)
    k += row[j] != undefined ? 1 : 0;
  var row = input.map(function(x) { return x * Math.pow(k+1, -0.5); });

  for (var layer = 0; layer < this.Ws.length; layer++) {
    // console.log(this.Ws[layer].length + ' * ' + this.Ws[layer][0].length);
    // Compute row * Ws[layer] + bs[layer]
    var nextRow = [];
    for (var b = 0; b < this.bs[layer].length; b++)
      nextRow.push(this.bs[layer][b]);
    
    for (var a = 0; a < this.Ws[layer].length; a++)
      for (var b = 0; b < this.Ws[layer][0].length; b++)
	nextRow[b] += row[a] * this.Ws[layer][a][b];

    if (layer < this.Ws.length - 1) {
      for (var b = 0; b < this.bs[layer].length; b++)
	nextRow[b] = Math.max(0, nextRow[b]); // relu
    } else {
      for (var b = 0; b < this.bs[layer].length; b++)
	nextRow[b] = 1.0 / (1 + Math.exp(-nextRow[b]));
    }
    row = nextRow
  }
  return row;
}

Autoencoder.prototype.getCdfs = function(row) {
  var row = this.getOutput(row);
  var cdfs = [];
  for (var i = 0; i < this.splits.length; i++) {
    var j = this.splits[i][0];
    var x = this.splits[i][1];
    while (cdfs.length <= j)
      cdfs.push({'xy': []});
    cdfs[j].xy.push({'x': x, 'y': row[i]});
  }
  return cdfs;
}

Autoencoder.prototype.getPdfs = function(row, points) {
  // We cheat a bit and define some smoothing
  // Let's approximate the CDF as a sum of sigmoids:
  // CDF = sum_i (row[i+1] - row[i]) * sigm((x' - x) / D)
  // Where i2 is a fractional version of i
  // Then take the derivative to get the PDF
  var cdfs = this.getCdfs(row);

  if (points == undefined)
    points = 100;

  var pdfs = [];

  for (var j = 0; j < row.length; j++) {
    pdfs.push({'xy': [], 'xyQuartile': []});

    for (var p = 0; p < points; p++) {
      var x = cdfs[j].xy[0].x + (cdfs[j].xy[cdfs[j].xy.length-1].x - cdfs[j].xy[0].x) * p / (points - 1);
      var y = 0.0;
      var yCdf = 0.0;

      for (var i = 0; i < cdfs[j].xy.length; i++) {
	var D = this.Ds[j][i];

	var xp = cdfs[j].xy[i].x;
	var d = cdfs[j].xy[i].y;
	if (i > 0) d -= cdfs[j].xy[i-1].y;
	var delta = (x - xp) / D;
	var s = 1.0 / (1 + Math.exp(-delta));
	y += d * s * (1 - s) / D;
	yCdf += d * s;
      }
      pdfs[j].xy.push({'x': x, 'y': y});
      
      if (yCdf > 0.25 && yCdf < 0.75) {
	pdfs[j].xyQuartile.push({'x': x, 'y': y});
      }
    }
  }
  return pdfs;
}
