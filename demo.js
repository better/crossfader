var model = RandomForest.deserialize(data);

var app = angular.module("myApp",[]);
app.controller('InputController', function($scope){

  var row = [];
  $scope.samples = [];
  for (var j = 0; j < headers.length; j++) {
    row.push(undefined);
    $scope.samples.push([]);
  }
  
  for (var i = 0; i < 1000; i++) {
    var sample = model.fill(row);
    for (var j = 0; j < row.length; j++)
      $scope.samples[j].push(sample[j]);
  }

  $scope.headers = headers.map(function(x) { return x.replace(/_/g, ' '); });
});

app.directive("renderChart",function(){
  return function(scope, element, attrs){
    var index = scope.$index;
    histogram(element, scope.samples[index]);
  }
})

function histogram(element, samples) {
  var margin = {top: 10, right: 30, bottom: 30, left: 30},
      width = element[0].offsetWidth,
      height = element[0].offsetHeight;
  
  var x = d3.scale.linear()
      .domain([d3.min(samples), d3.max(samples)])
      .range([0, width]);
  
  var data = d3.layout.histogram()
      .bins(x.ticks(40))
  (samples);
  
  var y = d3.scale.linear()
      .domain([0, d3.max(data, function(d) { return d.y; })])
      .range([height, 0]);
  
  var xAxis = d3.svg.axis()
      .scale(x)
      .orient("bottom");

  var svg = d3.select(element[0])
      .append('svg')
      .attr("width", width + margin.left + margin.right)
      .attr("height", height + margin.top + margin.bottom)
      .append("g");

  var bar = svg.selectAll(".bar")
      .data(data)
      .enter().append("g")
      .attr("class", "bar")
      .attr("transform", function(d) { return "translate(" + x(d.x) + "," + y(d.y) + ")"; });
  
  bar.append("rect")
    .attr("x", 1)
    .attr("width", x(data[0].dx) - 1)
    .attr("height", function(d) { return height - y(d.y); });
  
  svg.append("g")
    .attr("class", "x axis")
    .attr("transform", "translate(0," + height + ")")
    .call(xAxis);
}
