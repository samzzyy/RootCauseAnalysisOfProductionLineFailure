# CorrelationAnalysis

##Background
这是一个为生产线出现不良产品时而进行的根因分析，即发现导致高不良率的机台或机台组合，再细化到发现机台上的不良根因参数。
## Framework
### data
我们使用多个相关性指标先进行不良机台定位，Infomation value，卡方统计量，做出置信度由高到低的rank后，使用公式$$\sqrt{R_{1i}*R_{2i}}$$确定不良机台。
