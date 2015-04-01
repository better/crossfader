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

Autoencoder.prototype.getCdfs = function(row) {
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

  // Compute cdfs
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
