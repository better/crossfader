var random_forests = require('./random_forests'),
    readline = require('readline');

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

readStdin(function(data) {
  var model = random_forests(data, 100);
});
