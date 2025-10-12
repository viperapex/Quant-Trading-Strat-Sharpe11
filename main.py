# y_hat = model(x)
# orders = strategy(y_hat)
# exchange(orders)

# regression model => BTCUSDT => future log return

# Data and analysis libraries
import random
import numpy as np
from datetime import datetime, timedelta
import polars as pl

# machine learning libraires
import torch
import torch.nn as nn
import torch.optim as optim
import research

# visualisation
import altair as alt

# data sources
import binance

research.set_seed(42)

pl.Config.set_tbl_width_chars(200)
pl.Config.set_fmt_str_lengths(100)
pl.Config.set_tbl_cols(-1)

# Trading pair symbol
sym = 'BTCUSDT'
# Historical data window in days (e.g., 6 months)
hist_data_window = 7 * 4 * 6
# time horizon of time series (time interval)
time_interval = '1h'
# Max number of auto-regressive lags
max_lags = 4
# Forecast horizon in steps
forecast_horizon = 1
# Sharpe annualized rate (so it's independent of time frequency)
annualized_rate = research.sharpe_annualization_factor(time_interval, 365, 24)

binance.download_trades(sym, hist_data_window)

ts = research.load_ohlc_timeseries(sym, time_interval)
ts

research.load_timeseries(sym, time_interval, pl.col(
    'price').quantile(0.5).alias('price_median'))


research.plot_static_timeseries(ts, sym, 'close', time_interval)

alt.data_transformers.enable("vegafusion")
research.plot_dyn_timeseries(ts, sym, 'close', time_interval)

# Feature Engineering

price_time_series = pl.DataFrame({'price': [100.0, 120.0, 100.0]})
research.plot_column(price_time_series, 'price')

price_time_series.with_columns(
    pl.col('price').diff().alias('delta'),
    ((pl.col('price')-pl.col('price').shift()) /
     pl.col('price').shift()).alias('return'),
    (pl.col('price')/pl.col('price').shift()).log().alias('log_return'),
)

ts = ts.with_columns((pl.col(
    'close')/pl.col('close').shift(forecast_horizon)).log().alias('close_log_return'))
ts

target = 'close_log_return'
lr = pl.col(target)
ts = ts.with_columns(
    lr.shift(forecast_horizon * 1).alias(f'{target}_lag_1'),
    lr.shift(forecast_horizon * 2).alias(f'{target}_lag_2'),
    lr.shift(forecast_horizon * 3).alias(f'{target}_lag_3'),
    lr.shift(forecast_horizon * 4).alias(f'{target}_lag_4'),
)
ts


ts = research.add_lags(ts, target, max_lags, forecast_horizon)
ts


ts = ts.drop_nulls()


research.plot_distribution(ts, target, no_bins=100)


research.plot_distribution(ts, 'close', no_bins=100)

# Build Model


class LinearModel(nn.Module):
    def __init__(self, input_features):
        super(LinearModel, self).__init__()
        self.linear = nn.Linear(input_features, 1)

    def forward(self, x):
        return self.linear(x)


# Model Complexity
input_features = 1

linear_model = LinearModel(input_features)

research.print_model_info(linear_model, "Linear Model")
research.total_model_params(linear_model)

# y = w * x + b

# Split by time

features = ['close_log_return_lag_1']
target = 'close_log_return'
test_size = 0.25

len(ts)

int(len(ts) * test_size)

split_idx = int(len(ts) * (1-test_size))
split_idx


ts_train, ts_test = ts[:split_idx], ts[split_idx:]


ts_train


ts_test


X_train = torch.tensor(ts_train[features].to_numpy(), dtype=torch.float32)
X_test = ts_test[features].to_torch().float()
y_train = torch.tensor(ts_train[target].to_numpy(), dtype=torch.float32)
y_test = torch.tensor(ts_test[target].to_numpy(), dtype=torch.float32)


X_train

X_train.shape

y_train

y_train.shape

y_train = y_train.reshape(-1, 1)
y_train

y_train.shape

y_test = y_test.reshape(-1, 1)
y_test

research.timeseries_train_test_split(ts, features, target, test_size)

# Batch Gradient Descent

# hyperparameters
no_epochs = 1000 * 5
lr = 0.0005

# create model
model = LinearModel(len(features))
# loss function
criterion = nn.MSELoss()
# optimizer
optimizer = optim.Adam(model.parameters(), lr=lr)

print("\nTraining model...")

for epoch in range(no_epochs):
    # forward pass
    y_hat = model(X_train)
    loss = criterion(y_hat, y_train)

    # backward pass
    optimizer.zero_grad()   # 1. clear old gradients
    loss.backward()         # 2. compute new gradients
    optimizer.step()        # 3. update weights

    # check for improvement
    train_loss = loss.item()

    # logging
    if (epoch + 1) % 500 == 0:
        print(f"Epoch [{epoch+1}/{no_epochs}], Loss: {train_loss:.6f}")

print("\nLearned parameters")

for name, param in model.named_parameters():
    if param.requires_grad:
        print(f"{name}:\n{param.data.numpy()}")

# Evaluation
model.eval()
with torch.no_grad():
    y_hat = model(X_test)
    test_loss = criterion(y_hat, y_test)
    print(f"\nTest Loss: {test_loss.item():.6f}, Train Loss: {train_loss:.6f}")


# Test Trading Peformance

trade_results = pl.DataFrame({
    'y_hat': y_hat.squeeze(),
    'y': y_test.squeeze()
}).with_columns(
    (pl.col('y_hat').sign() == pl.col('y').sign()).alias('is_won'),
    pl.col('y_hat').sign().alias('signal'),
).with_columns(
    (pl.col('signal') * pl.col('y')).alias('trade_log_return')
).with_columns(
    pl.col('trade_log_return').cum_sum().alias('equity_curve')
)
trade_results

research.plot_column(trade_results, 'equity_curve')

trade_results = trade_results.with_columns(
    (pl.col('equity_curve')-pl.col('equity_curve').cum_max()).alias('drawdown_log')
)
trade_results

max_drawdown_log = trade_results['drawdown_log'].min()
max_drawdown_log

drawdown_pct = np.exp(max_drawdown_log) - 1
drawdown_pct

equity_peak = 1000
equity_peak * drawdown_pct

win_rate = trade_results['is_won'].mean()
win_rate

avg_win = trade_results.filter(pl.col('is_won') == True)[
    'trade_log_return'].mean()
avg_loss = trade_results.filter(pl.col('is_won') == False)[
    'trade_log_return'].mean()
ev = win_rate * avg_win + (1 - win_rate) * avg_loss
ev

total_log_return = trade_results['trade_log_return'].sum()
total_log_return

compound_return = np.exp(total_log_return)
compound_return

1000*compound_return

equity_trough = trade_results['equity_curve'].min()
equity_trough

equity_peak = trade_results['equity_curve'].max()
equity_peak

std = trade_results['trade_log_return'].std()
std

sharpe = ev / std * annualized_rate
sharpe

research.eval_model_performance(
    y_test, y_hat, features, target, annualized_rate)

target = 'close_log_return'
features = ['close_log_return_lag_2']
model = LinearModel(len(features))
perf = research.benchmark_reg_model(
    ts, features, target, model, annualized_rate, no_epochs=50)
