Autoencoder = function(headers, splits, Ws, bs) {
  this.headers = headers;
  this.splits = splits;
  this.Ws = Ws;
  this.bs = bs;
}

Autoencoder.deserialize = function(data) {
  return new Autoencoder(data['headers'],
			 data['splits'],
			 data['Ws'],
			 data['bs']);
}

Autoencoder.prototype.getOutput = function(row) {
  var input = [];
  for (var i = 0; i < this.splits.length; i++) {
    var j = this.splits[i][0];
    var x = this.splits[i][1];
    if (row[j] == undefined)
      input.push(0.0);
    else if (row[j] < x)
      input.push(1.0);
    else if (row[j] > x)
      input.push(-1.0);
  }
  var row = input;
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
      cdfs.push([]);
    cdfs[j].push({'x': x, 'y': row[i]});
  }
  return cdfs;
}

Autoencoder.prototype.getPdfs = function(row, points) {
  // We cheat a bit and define some smoothing
  // Let's approximate the CDF as a sum of sigmoids:
  // CDF = sum_i (row[i+1] - row[i]) * sigm(i2 - i)
  // Where i2 is a fractional version of i
  // Then take the derivative to get the PDF
  var cdfs = this.getCdfs(row);

  if (points == undefined)
    points = 100;

  var pdfs = [];

  for (var j = 0; j < row.length; j++) {
    pdfs.push([]);
    // console.log(cdfs[j]);
    for (var p = 0; p < points; p++) {
      var x = cdfs[j][0].x + (cdfs[j][cdfs[j].length-1].x - cdfs[j][0].x) * p / (points - 1);
      var i = p / (points - 1) * (cdfs[j].length); // todo: assumes even spacing
      var y = 0.0;

      for (var i2 = 0; i2 < cdfs[j].length; i2++) {
	var d = cdfs[j][i2].y;
	if (i2 > 0) d -= cdfs[j][i2-1].y;
	var s = 1.0 / (1 + Math.exp(i2 - i));
	y += d * s * (1 - s);
	// console.log(i2 + ' ' + d + ' ' + s);
      }
      pdfs[j].push({'x': x, 'y': y});
    }
  }
  return pdfs;
}
