const topNProductsChart = echarts.init(document.getElementById('top_n_products-chart'));

const option = {
  title: {
    text: 'Top 5 Products by Sales'
  },
  tooltip: {
    trigger: 'axis',
    axisPointer: {
      type: 'shadow'
    }
  },
  xAxis: {
    type: 'category',
    data: ["DOTCOM POSTAGE", "REGENCY CAKESTAND 3 TIER", "WHITE HANGING HEART T-LIGHT HOLDER", "PARTY BUNTING", "JUMBO BAG RED RETROSPOT"]
  },
  yAxis: {
    type: 'value'
  },
  series: [
    {
      type: 'bar',
      data: [206245.48, 164762.19, 99668.47, 98302.98, 92356.03]
    }
  ]
};

topNProductsChart.setOption(option);