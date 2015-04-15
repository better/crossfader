var app = angular.module("myApp",[]);
app.controller('ChartsController', ChartsController);

var examples = {};

function ChartsController($scope) {
  $scope.headers = [];
  $scope.model = [];
  $scope.examples = ['stocks', 'countries', 'wine', 'car'];

  $scope.setData = function(key) {
    if (window.location.protocol == 'file:')
      // We can't load a JS dynamically from local disk, so if it's local then load it from Github
      var url = 'https://rawgit.com/bettermg/crossfader/master/examples/models/' + key + '.js';
    else
      var url = window.location.href.split('demo.html')[0] + 'examples/models/' + key + '.js';
    console.log('url: ' + url);

    if (examples[key])
      $scope._setData(examples[key]);
    else
      $.getScript(url, function() {
	      $scope._setData(examples[key]);
	      $scope.$apply();
	  });
  }

  $scope._setData = function(example) {
    $scope.charts = [];
    $scope.model = Autoencoder.deserialize(example);
    $scope.headers = example.headers;
  };

  $scope.init = function() {
    var key = window.location.hash.substring(1);
    if (key) {
      $scope.setData(key);
    }
  }
}

app.directive("renderChart",function(){
  return function($scope, element, attrs){
    var j = $scope.$index;
    var getUpdateFn = function(index) {
      return function(update, newValue) { redraw($scope.model, $scope.charts, update, index, newValue); };
    }
    var chart = new Chart(element, getUpdateFn(j));

    $scope.charts.push(chart);
    
    if ($scope.charts.length == $scope.headers.length) {
      redraw($scope.model, $scope.charts, true);
    }
  }
});

function redraw(model, charts, update, index, newValue) {
  if (update && newValue != undefined)
    charts[index].fixedValue = newValue;

  var row = [];
  for (var j = 0; j < charts.length; j++)
    row.push(charts[j].fixedValue);

  if (newValue != undefined)
    row[index] = newValue;

  var curves = model.getPdfs(row);

  for (var j = 0; j < row.length; j++) {
    if (update) {
      charts[j].update(curves[j]);
      charts[j].render(null);
    } else {
      charts[j].render(curves[j]);
    }
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
      .attr("width", this.width) // + this.margin.left + this.margin.right)
      .attr("height", this.height) // - this.margin.top - this.margin.bottom)
      .append("g");

  this.axis = svg.append("g")
    .attr("class", "x axis")
    .attr("transform", "translate(0," + (this.height - this.margin.bottom) + ")");

  this.path = svg.append('path')
    .attr('stroke', 'black')
    .attr('stroke-weight', '5')
    .attr('fill', 'none');

  this.hypoPath = svg.append('path')
    .attr('stroke', 'green')
    .attr('stroke-weight', '5')
    .attr('fill', 'none');

  this.hypoPathQuartiles = svg.append('path')
    .attr('fill', 'green')
    .attr('fill-opacity', '0.1');

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
    .on("mouseover", move)
    .on("mouseout", out)
    .on("mousemove", move)
    .on("click", click)

  var bisectData = d3.bisector(function(d) { return d.x; }).left;

  var chart = this;

  function click() {
    focus.locked = true;

    var x0 = chart.x.invert(d3.mouse(this)[0]);
    
    focus.attr("transform", "translate(" + chart.x(x0) + "," + chart.y(0) + ")");
    focus.select("text").text(x0.toPrecision(4));
    
    chart.redraw(true, x0);
  }

  function move() {
    if (!chart.data)
      return;

    if (focus.locked)
      return;

    focus.style("display", null);
    
    var x0 = chart.x.invert(d3.mouse(this)[0]);
      
    focus.attr("transform", "translate(" + chart.x(x0) + "," + chart.y(0) + ")");
    focus.select("text").text(x0.toPrecision(4));
    
    chart.redraw(false, x0);
  }

  function out() {
    if (!focus.locked)
      focus.style("display", 'none');

    chart.redraw(true, null);
  }
}

Chart.prototype.update = function(data) {
  this.data = data;
}

Chart.prototype.render = function(hypoData) {  
  this.x = d3.scale.linear()
    .domain(d3.extent(this.data.xy, function(d) { return d.x; }))
    .range([0, this.width]);

  var getY = function(d) { return d.y };
  var max = d3.max(this.data.xy, getY);
  if (hypoData)
      max = Math.max(max, d3.max(hypoData.xy, getY));

  this.y = d3.scale.linear()
    .domain([0, max])
    .range([this.height - this.margin.bottom, 0]);

  var xAxis = d3.svg.axis()
      .scale(this.x)
      .ticks(5)
      .orient("bottom");

  this.axis.call(xAxis);

  var chart = this;
  var lineFunction = d3.svg.line()
      .x(function(d) { return chart.x(d.x); })
      .y(function(d) { return chart.y(d.y); });

  this.path.attr('d', lineFunction(this.data.xy));

  if (hypoData == null) {
    this.hypoPath.attr('display', 'none');
    this.hypoPathQuartiles.attr('display', 'none');

  } else {
    this.hypoPath.attr('d', lineFunction(hypoData.xy));
    this.hypoPath.attr('display', null);

    if (hypoData.xyQuartile && hypoData.xyQuartile.length > 0) {
      hypoData.xyQuartile.push({'x': hypoData.xyQuartile[hypoData.xyQuartile.length-1].x, 'y': 0});
      hypoData.xyQuartile.push({'x': hypoData.xyQuartile[0].x, 'y': 0});

      this.hypoPathQuartiles.attr('d', lineFunction(hypoData.xyQuartile));
      this.hypoPathQuartiles.attr('display', null);
    }
  }
}
