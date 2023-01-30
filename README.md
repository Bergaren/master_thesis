# Master Thesis Project @ Krafthem AB

Project Report: [Link to overleaf](https://www.overleaf.com/read/pmtmmrygqznb)

Commonly, the day-ahead and intraday market on the electricity exchange are treated separately in academia. However, a model that forecasts the direction of the price spread between these two markets creates an opportunity for a market participant to leverage the price spread. In the neighbouring domain, electricity price forecasting, deep learning has shown to excel. Therefore, it is hypothesised that it will do so in directional price spread forecasting as well. A quantitative case study was performed to investigate how accurately a deep learning approach could be in directional electricity price spread forecasting. The case study was conducted on the Nordic electricity exchange Nord Pool in the SE3 region. The deep learning approach was compared with previously suggested machine learning models and a naive heuristic. The results show no statistical difference in error rate between the deep learning model and the machine learning model or naive heuristic.
The results suggest that deep learning might not be a suitable approach to the task or that the implementation did not fully exhaust the potential of deep learning.

## Structure

```
├── models.py
├── main.py
├── dataset.py
├── benchmark.py
├── config.py
└── .env // Create and add EQ API key here
```

```
models/
├── Saved models
└── Saved hyperparameters
```

```
results/
├── misc/
│   └── figures created for report
├── Performance on test and validation set
└── Output from runs on test and validation set
```

```
data/
├── consumption data/
│   └── empty (licensed data)
├── production_data/
│   └── empty (licensed data)
├── price_yearly/
│   └── elspot & elbas price data for each year
├── elspot_dayahead/
│   └── elspot data for each year
├── price_data.csv
└── data.py
```