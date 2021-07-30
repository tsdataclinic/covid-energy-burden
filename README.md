# Estimating additional energy use during Covid

With the move to learning and working from home and other measures such as stay-at-home orders during the Covid-19 pandemic, there was a shift in energy consumption patterns. In this project, we try to estimate the effect of the pandemic on residential and commercial energy consumption. While aggregate measures of energy use give us an idea of the overall change due to the pandemic, we have tried to refine this estimate by developing a model for energy consumption over time and using it to refine the estimate of the increase or decrease in usage across the different mainland US states. 

## Methodology

The key idea in this approach is to compare the actual energy consumption against expected consumption had there not been a pandemic. We estimate the latter measure by developing a model for energy use that account for the population, seasonality, trends over time, and weather-related variables. The model performance in the years prior to the pandemic indicated that it tracks actual energy use fairly well, allowing us to use it to generate predictions starting in 2020. These are the estimates of what energy use would have looked like had trends from the years prior continued.Lastly, we compare these predictions against the actual energy use during the pandemic to report the differences [here](data/energy_data_with_predictions_v2). 

We used the [FBProphet](https://facebook.github.io/prophet/) library to fit a Bayesian additive model for forecasting the energy use time-series.

## Data Sources

Data from the US Energy Information Administration (EIA):
[Monthly energy sales by state and sector](https://www.eia.gov/opendata/qb.php?category=38)
[Number of consumers](https://www.eia.gov/opendata/qb.php?category=1718389)
[Monthly Heating and cooling days](https://www.eia.gov/opendata/qb.php?category=829723)

Data from the National Oceanic and Atmospheric Administration (NOAA):
[Storm events](https://www1.ncdc.noaa.gov/pub/data/swdi/stormevents/csvfiles/)
