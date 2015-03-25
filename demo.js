var model = RandomForest.deserialize(data);

var app = angular.module("myApp",[]);
app.controller('InputController', function($scope){

  $scope.row = [];
  for (var j = 0; j < headers.length; j++)
    $scope.row.push(undefined);

  $scope.headers = headers.map(function(x) { return x.replace(/_/g, ' '); });
  $scope.charts = [];
});

app.directive("renderChart",function(){
  return function($scope, element, attrs){
    var chart = new Chart(element);
    $scope.charts.push(chart);
    
    if ($scope.charts.length == $scope.row.length) {
      redraw($scope.charts, $scope.row);
    }
  }
});

function redraw(charts, row) {
  var samples = [];
  for (var j = 0; j < row.length; j++)
    samples.push([]);

  for (var i = 0; i < 1000; i++) {
    var sample = model.fill(row);
    for (var j = 0; j < row.length; j++)
      samples[j].push(sample[j]);
  }

  for (var j = 0; j < row.length; j++) {
    console.log('redrawing ' + j);
    charts[j].update(samples[j]);
  }
}

function Chart(element) {
  this.margin = {top: 10, right: 30, bottom: 30, left: 30};
  this.width = element[0].offsetWidth,
  this.height = element[0].offsetHeight,
  this.nBins = 30;

  var barWidth = Math.floor(this.width/this.nBins);
  this.width = barWidth * this.nBins;
  
  var svg = d3.select(element[0])
      .append('svg')
      .attr("width", this.width + this.margin.left + this.margin.right)
      .attr("height", this.height + this.margin.top + this.margin.bottom)
      .append("g");

/*  svg.append("g")
    .attr("class", "x axis")
    .attr("transform", "translate(0," + (height-margin.bottom) + ")")
    .call(xAxis);*/

  this.path = svg.append('path')
    .attr('stroke', 'black')
    .attr('stroke-weight', '5')
    .attr('fill', 'none');

  /*var focus = svg.append("g")
      .attr("class", "focus")
      .style("display", "none");
  
  focus.append("circle")
    .attr("r", 4.5);
  
  focus.append("text")
    .attr("x", 9)
    .attr("dy", ".35em");

  svg.append("rect")
    .attr("fill", "none")
    .attr("pointer-events", "all")
    .attr("width", width)
    .attr("height", height)
    .on("mouseover", function() { focus.style("display", null); })
    .on("mouseout", function() { focus.style("display", "none"); })
    .on("mousemove", mousemove);


  this.bisectData = d3.bisector(function(d) { return d.x; }).left;
  
  function mousemove() {
    var x0 = x.invert(d3.mouse(this)[0]),
	i = bisectData(data, x0, 1),
	d = data[i];
    // TODO: linear interpolateation 
    focus.attr("transform", "translate(" + x(d.x) + "," + y(d.y) + ")");
    focus.select("text").text(d.x);
    setValue(d.x);
  }*/
}

Chart.prototype.update = function(samples) {
 var x = d3.scale.linear()
      .domain([d3.min(samples), d3.max(samples)])
      .range([0, this.width]);
  
  var bins = d3.layout.histogram()
      .bins(x.ticks(this.nBins));

  var data = bins(samples);

  var y = d3.scale.linear()
      .domain([0, d3.max(data, function(d) { return d.y; })])
      .range([this.height, 0]);

  var xAxis = d3.svg.axis()
      .scale(x)
      .orient("bottom");

  var lineFunction = d3.svg.line()
      .x(function(d) { return x(d.x); })
      .y(function(d) { return y(d.y); })
      .interpolate("basis");

  this.path.attr('d', lineFunction(data))
}

