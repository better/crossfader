var model = RandomForest.deserialize(data);

var app = angular.module("myApp",[]);
app.controller('InputController', function($scope){
  $scope.headers = headers.map(function(x) { return x.replace(/_/g, ' '); });
  $scope.charts = [];
});

app.directive("renderChart",function(){
  return function($scope, element, attrs){
    var chart = new Chart(element, function() { redraw($scope.charts); });
    $scope.charts.push(chart);
    
    if ($scope.charts.length == $scope.headers.length) {
      redraw($scope.charts);
    }
  }
});

function redraw(charts) {
  var row = [];
  for (var j = 0; j < charts.length; j++)
    row.push(charts[j].fixedValue);

  console.log(row);

  var samples = [];
  for (var j = 0; j < row.length; j++)
    samples.push([]);

  for (var i = 0; i < 1000; i++) {
    var sample = model.fill(row);
    for (var j = 0; j < row.length; j++)
      samples[j].push(sample[j]);
  }

  for (var j = 0; j < row.length; j++) {
    charts[j].update(samples[j]);
  }
}

function Chart(element, redraw) {
  this.margin = {top: 10, right: 30, bottom: 30, left: 30};
  this.width = element[0].offsetWidth,
  this.height = element[0].offsetHeight,
  this.nBins = 30;
  this.fixedValue = undefined;
  this.redraw = redraw;

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

  var focus = svg.append("g")
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
    .attr("width", this.width)
    .attr("height", this.height)
    .on("mouseover", function() { focus.style("display", null); })
    .on("mouseout", function() { focus.style("display", "none"); })
    .on("mousemove", getEventHandler(false))
    .on("click", getEventHandler(true));

  var bisectData = d3.bisector(function(d) { return d.x; }).left;

  var chart = this;

  function getEventHandler(update) {
    return function() {
      if (!chart.data)
	return;
    
      var x0 = chart.x.invert(d3.mouse(this)[0]),
	  i = bisectData(chart.data, x0, 1),
	  d = chart.data[i];
      
      // TODO: linear interpolation
      focus.attr("transform", "translate(" + chart.x(d.x) + "," + chart.y(d.y) + ")");
      focus.select("text").text(x0);
      
      if (update) {
	chart.fixedValue = x0;
	chart.redraw();
      }
    }
  }
}

Chart.prototype.update = function(samples) {
  this.x = d3.scale.linear()
    .domain([d3.min(samples), d3.max(samples)])
    .range([0, this.width]);
  
  var bins = d3.layout.histogram()
      .bins(this.x.ticks(this.nBins));

  this.data = bins(samples);

  this.y = d3.scale.linear()
    .domain([0, d3.max(this.data, function(d) { return d.y; })])
    .range([this.height, 0]);

  var xAxis = d3.svg.axis()
      .scale(this.x)
      .orient("bottom");

  var chart = this;
  var lineFunction = d3.svg.line()
      .x(function(d) { return chart.x(d.x); })
      .y(function(d) { return chart.y(d.y); })
      .interpolate("basis");

  this.path.attr('d', lineFunction(this.data))
}

