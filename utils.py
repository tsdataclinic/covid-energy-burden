import os
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import mean_absolute_error as MAE
from datetime import datetime
from fbprophet import Prophet

def median_filter(df, varname = 'y', window=24, std=2.75): 
    """
    A simple median filter, removes (i.e. replace by np.nan) observations that exceed N (default = 3) 
    tandard deviation from the median over window of length P (default = 24) centered around 
    each observation.
    Parameters
    ----------
    df : pandas.DataFrame
        The pandas.DataFrame containing the column to filter.
    varname : string
        Column to filter in the pandas.DataFrame. No default. 
    window : integer 
        Size of the window around each observation for the calculation 
        of the median and std. Default is 24 (time-steps).
    std : integer 
        Threshold for the number of std around the median to replace 
        by `np.nan`. Default is 3 (greater / less or equal).
    Returns
    -------
    dfc : pandas.Dataframe
        A copy of the pandas.DataFrame `df` with the new, filtered column `varname`
    """
    
    dfc = df.loc[:,[varname]]
    
    dfc['median']= dfc[varname].rolling(window, center=True).median()
    
    dfc['std'] = dfc[varname].rolling(window, center=True).std()
    
    dfc.loc[dfc.loc[:,varname] >= dfc['median']+std*dfc['std'], varname] = np.nan
    
    dfc.loc[dfc.loc[:,varname] <= dfc['median']-std*dfc['std'], varname] = np.nan
    
    return dfc.loc[:, varname]

def prepare_data(data, year=2017): 
    """
    prepare the data for ingestion by fbprophet: 
    see: https://facebook.github.io/prophet/docs/quick_start.html
    
    1) divide in training and test set, using the `year` parameter (int)
    
    2) reset the index and rename the `datetime` column to `ds`
    
    returns the training and test dataframes
    Parameters
    ----------
    data : pandas.DataFrame 
        The dataframe to prepare, needs to have a datetime index
    year: integer 
        The year separating the training set and the test set (includes the year)
    Returns
    -------
    data_train : pandas.DataFrame
        The training set, formatted for fbprophet.
    data_test :  pandas.Dataframe
        The test set, formatted for fbprophet.
    """
    
    
    data_train = data[:str(year-1)]
    
    data_test = data[str(year):]
    
    data_train.reset_index(inplace=True)
    
    data_test.reset_index(inplace=True)
    
    data_train = data_train.rename({'date':'ds'}, axis=1)
    
    data_test = data_test.rename({'date':'ds'}, axis=1)
    
    return data_train, data_test


def add_regressor(data, regressor, varname=None): 
    
    """
    adds a regressor to a `pandas.DataFrame` of target (predictand) values 
    for use in fbprophet 
    Parameters
    ----------
    data : pandas.DataFrame 
        The pandas.DataFrame in the fbprophet format (see function `prepare_data` in this package)
    regressor : pandas.DataFrame 
        A pandas.DataFrame containing the extra-regressor
    varname : string 
        The name of the column in the `regressor` DataFrame to add to the `data` DataFrame
    Returns
    -------
    verif : pandas.DataFrame
        The original `data` DataFrame with the column containing the 
        extra regressor `varname`
    """

    data_with_regressors = data.copy()
    
    data_with_regressors.loc[:,varname] = regressor.loc[:,varname]
    
    return data_with_regressors

def add_regressor_to_future(future, regressors): 
    """
    adds extra regressors to a `future` DataFrame dataframe created by fbprophet
    Parameters
    ----------
    data : pandas.DataFrame
        A `future` DataFrame created by the fbprophet `make_future` method  
        
    regressors_df: pandas.DataFrame 
        The pandas.DataFrame containing the regressors (with a datetime index)
    Returns
    -------
    futures : pandas.DataFrame
        The `future` DataFrame with the regressors added
    """
    
    futures = future.copy() 
    
    futures.index = pd.to_datetime(futures.ds)
    
    futures = futures.merge(regressors, left_index=True, right_index=True)
    
    futures = futures.reset_index(drop = True)
    
    return futures


def make_verif(forecast, data_train, data_test): 
    """
    Put together the forecast (coming from fbprophet) 
    and the overved data, and set the index to be a proper datetime index, 
    for plotting
    Parameters
    ----------
    forecast : pandas.DataFrame 
        The pandas.DataFrame coming from the `forecast` method of a fbprophet 
        model. 
    
    data_train : pandas.DataFrame
        The training set, pandas.DataFrame
    data_test : pandas.DataFrame
        The training set, pandas.DataFrame
    
    Returns
    -------
    forecast : 
        The forecast DataFrane including the original observed data.
    """
    
    forecast.index = pd.to_datetime(forecast.ds)
    
    data_train.index = pd.to_datetime(data_train.ds)
    
    data_test.index = pd.to_datetime(data_test.ds)
    
    data = pd.concat([data_train, data_test], axis=0)
    
    forecast.loc[:,'y'] = data.loc[:,'y']
    
    return forecast

def plot_verif(verif, year=2017):
    """
    plots the forecasts and observed data, the `year` argument is used to visualise 
    the division between the training and test sets. 
    Parameters
    ----------
    verif : pandas.DataFrame
        The `verif` DataFrame coming from the `make_verif` function in this package
    year : integer
        The year used to separate the training and test set. Default 2017
    Returns
    -------
    f : matplotlib Figure object
    """
    
    f, ax = plt.subplots(figsize=(14, 8))
    
    train = verif.loc[:str(year - 1),:]
    
    ax.plot(train.index, train.y, 'ko', markersize=3)
    
    ax.plot(train.index, train.yhat, lw=0.5)
    
    ax.fill_between(train.index, train.yhat_lower, train.yhat_upper, alpha=0.3)
    
    test = verif.loc[str(year):,:]
    
    ax.plot(test.index, test.y, 'ro', markersize=3)
    
    ax.plot(test.index, test.yhat, lw=0.5)
    
    ax.fill_between(test.index, test.yhat_lower, test.yhat_upper, alpha=0.3)
    
    ax.axvline(datetime(year,1,1), color='0.8', alpha=0.7)
    
    ax.grid(ls=':', lw=0.5)
    
    return f

def plot_verif_component(verif, component='rain', year=2017): 
    """
    plots a specific component of the `verif` DataFrame
   Parameters
    ----------
    verif : pandas.DataFrame
        The `verif` DataFrame coming from the `make_verif` function in this package. 
    component : string 
        The name of the component (i.e. column name) to plot in the `verif` DataFrame. 
    year : integer
        The year used to separate the training and test set. Default 2017
    Returns
    -------
    f : matplotlib Figure object
    """
    
    f, ax = plt.subplots(figsize=(14, 7))
    
    train = verif.loc[:str(year - 1),:]
        
    ax.plot(train.index, train.loc[:,component] * 100, color='0.8', lw=1, ls='-')
    
    ax.fill_between(train.index, train.loc[:, component+'_lower'] * 100, train.loc[:, component+'_upper'] * 100, color='0.8', alpha=0.3)
    
    test = verif.loc[str(year):,:]
        
    ax.plot(test.index, test.loc[:,component] * 100, color='k', lw=1, ls='-')
    
    ax.fill_between(test.index, test.loc[:, component+'_lower'] * 100, test.loc[:, component+'_upper'] * 100, color='0.8', alpha=0.3)
    
    ax.axvline(str(year), color='k', alpha=0.7)
    
    ax.grid(ls=':', lw=0.5)
    
    return f


def plot_joint_plot(verif, x='yhat', y='y', title=None, fpath = '../figures/paper', fname = None): 
    """
    
    Parameters
    ---------- 
    verif : pandas.DataFrame 
    x : string 
        The variable on the x-axis
        Defaults to `yhat`, i.e. the forecast or estimated values.
    y : string 
        The variable on the y-axis
        Defaults to `y`, i.e. the observed values
    title : string 
        The title of the figure, default `None`. 
    
    fpath : string 
        The path to save the figures, default to `../figures/paper`
    fname : string
        The filename for the figure to be saved
        ommits the extension, the figure is saved in png, jpeg and pdf
 
    Returns
    -------
    f : matplotlib Figure object
    """

    g = sns.jointplot(x='yhat', y='y', data = verif, kind="reg", color="0.4")
    
    g.fig.set_figwidth(8)
    g.fig.set_figheight(8)

    ax = g.fig.axes[1]
    
    if title is not None: 
        ax.set_title("R = {:+4.2f}\nMAE = {:4.1f}".format(verif.loc[:,['y','yhat']].corr().iloc[0,1], MAE(verif.loc[:,'y'].values, verif.loc[:,'yhat'].values)), fontsize=16)

    ax = g.fig.axes[0]

    ax.set_xlabel("model's estimates", fontsize=15)
    
    ax.set_ylabel("observations", fontsize=15)
    
    ax.grid(ls=':')

    [l.set_fontsize(13) for l in ax.xaxis.get_ticklabels()]
    [l.set_fontsize(13) for l in ax.yaxis.get_ticklabels()];

    ax.grid(ls=':')
    
    if fname is not None: 
        for ext in ['png','jpeg','pdf']: 
            g.fig.savefig(os.path.join(fpath, "{}.{}".format(fname, ext)), dpi=200)

            
def state_plot(df, state, sector,col='pct_error', year_lim=2019):
    df_plot = df[(df.state==state)&(df.sector==sector)&(df.year>=year_lim)].copy()
    fig,ax = plt.subplots(figsize=(10,4))
    mean_error_before = df_plot[(df_plot.date<'2020-03-01')][col].mean()
    mean_error_after = df_plot[(df_plot.date>='2020-03-01')][col].mean()
    
#     df_plot = df_plot.set_index('date').sort_index()
#     df_plot[col].plot(ax=ax, marker="o")
    df_plot = df_plot.sort_values('date')
    plot = sns.lineplot(x='date',y=col,data=df_plot,ax=ax, marker='o', markersize=7)
    x_dates = [t.strftime('%b\n%Y') if t.month==1 else t.strftime('%b') for t in df_plot[df_plot.date.dt.month%3 ==1]['date']]
    ax.set_xticklabels(labels=x_dates)
# #     for ind, label in enumerate(plot.get_xticklabels()):
#         if ind % 3 == 0:  # every 10th label is kept
#             label.set_visible(True)
#         else:
#             label.set_visible(False)
    
    plt.xlabel('Month')
    plt.ylabel('% Error in prediction')
    plt.axvline(x=datetime(2020,2,15),color='#f76d23',linestyle='dotted')
    plt.axhline(y=mean_error_before, xmin=0, xmax=0.63, color='r', linestyle='--',linewidth=1.5)
    plt.axhline(y=mean_error_after, xmin=0.63, xmax=1, color='g', linestyle='--',linewidth=1.5)
    sign = '+' if mean_error_after-mean_error_before > 0 else '-'
    text_color = 'g' if mean_error_after-mean_error_before > 0 else 'r'
    plt.text(x=datetime(2020,2,10),y=(mean_error_after),s= "{}{}%".format(sign, np.round(mean_error_after-mean_error_before,2)),
             fontsize=13,horizontalalignment='right', color=text_color, fontweight='bold')
    plt.title("Prediction error over time for {} sector in {}".format(sector,state))
#     plt.axvline()


def get_model_for_state_sector(data, state, sector, split_year=2019, plot_forecast=False, changepoint_prior_scale=0.5):
    ## Defining Training Data 
    df_model = data[(data.state == state)&(data.sector == sector)].copy().set_index('date').sort_index()
    df_train, df_test = prepare_data(df_model[['y','heating_days','cooling_days','pct_weekdays']], year=split_year)
    regressors_df = df_model[['heating_days','cooling_days','pct_weekdays']]
    ## Defining Prophet Model
    m = Prophet(seasonality_mode='multiplicative',
                yearly_seasonality=5,daily_seasonality=False,weekly_seasonality=False,mcmc_samples=300,
                changepoint_prior_scale=changepoint_prior_scale, changepoint_range=0.95)
    m.add_regressor('heating_days', mode='additive')
    m.add_regressor('cooling_days', mode='additive')
    m.add_regressor('pct_weekdays', mode='additive')
    # m.add_regressor('weird_range', mode='additive')
    m_fit = m.fit(df_train,control={'max_treedepth': 12})
    ## Getting forecasts
    future = m_fit.make_future_dataframe(periods = 21, freq = 'MS')
    future = add_regressor_to_future(future, regressors_df)
    # future = future.merge(df_model[['date','heating_days','cooling_days','pct_weekdays']],how='left',left_on='ds',right_on='date').drop(columns=['date']).dropna()
    forecast = m_fit.predict(future)
    if plot_forecast:
        fig = m_fit.plot(forecast)
    m.plot_components(forecast)
    ## Validation
    verif = make_verif(forecast[forecast.ds.dt.year<=2020], df_train, df_test)
    print("Prediction Correlation: {}".format(verif.loc[:,['y','yhat']].corr()['y']['yhat']))
    if plot_forecast:
        f =  plot_verif(verif,split_year)
        plot_joint_plot(verif.loc['2019':'2019',:], title='test set')