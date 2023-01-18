import pandas as pd
from pandas_datareader import data as pdr
import yfinance as yf
import datetime as dt
import numpy as np
import plotly.graph_objects as go
import os
from pandas.tseries.offsets import MonthEnd
from plotly.subplots import make_subplots
import math
import argparse



pd.options.plotting.backend = "plotly"


class YahooCsv:
    def __init__(self, input_file):
        self.input_file = input_file

    def load(self):
        pf = pd.read_csv(self.input_file)
        pf["Trade Date"] = pd.to_datetime(pf["Trade Date"], format="%Y%m%d")
        # A quick pivot table to get the list of movements and work with that
        movements = pf.pivot_table(
            values=pf.columns,
            index=["Symbol", "Trade Date"],
            aggfunc={"Purchase Price": np.average, "Quantity": sum},
        )
        tickers = movements.index.get_level_values("Symbol").unique()
        # Let's slice the multi-index table and build our positions
        positions = {}
        for t in tickers:
            print("Loading %s ..." % t)
            p = movements.xs(t)
            p["Symbol"] = t
            positions[t] = Position(t, p)

        return positions


class Stock:
    # Base class for any stock
    def __init__(self, ticker, start_date):
        self.ticker = ticker
        self.prices = self._load_prices(start_date)
        # The only data we have for a stock is its adjusted closed price
        self.df = pd.concat([self.prices], axis=1)
        self._populate()
        self.current_price = round(self.prices.iloc[-1], 2)
        self.profit = round(self.df['Profit/Loss'].iloc[-1], 2)

    def _load_prices(self, start_date):
        # Check first if we already have some historical prices saved in cache
        # start_date needs to be a datetime
        yf.pdr_override()
        cache = Cache(self.ticker + ".pkl")
        cached_prices = cache.load()
        if cache.found and cache.start_date() < start_date:
            start_date = cache.end_date() + dt.timedelta(days=1)
        new_data = False
        today = dt.datetime.today()
        print(today, start_date)
        if today > start_date:
            # Let's avoid making calls to Yahoo if not needed
            new_prices = pdr.get_data_yahoo([self.ticker], start=start_date)["Adj Close"]
            new_prices.index = new_prices.index.tz_localize(None)
            new_data = True
        if cache.found and cache.start_date() < start_date and new_data:
            prices = pd.concat([cached_prices, new_prices])
            # Yahoo tends to return at least one value so we need to deduplicate it just in case
            prices = prices[~prices.index.duplicated(keep='first')]
        elif new_data:
            prices = new_prices
        elif cache.found:
            prices = cached_prices
        # dropping dup index rows
        cache.save(prices)
        return prices

    def _populate(self):
        # A basic stock has only information about prices and
        self.df['Profit/Loss'] = self.df['Adj Close'] - self.df.iloc[0]['Adj Close']
        self.df['Profit/Loss (%)'] = (self.df['Profit/Loss'] / self.df.iloc[0]['Adj Close']) * 100
        # Reindexing over the full year with all the days to take into account for market closed days
        start_date = self.df.index.min()
        end_date = self.df.index.max()
        period = pd.date_range(start_date, end_date)
        # sometimes it seems that Yahoo Finance returns duplicated rows... deduplicating here
        self.df = self.df[~self.df.index.duplicated(keep='first')]
        self.df = self.df.reindex(period, method='ffill')

    def _rebase_profits_from(self, start_date):
        # Recalculate the profits from given date
        self.df = self.df.loc[start_date:]
        self._populate()

    @classmethod
    def plot_this(self, df, labels):
        # df is a dataframe containing an arbitrary amount of columns
        # labels is a dict containing title, xaxis label, yaxis label
        # we use the classmethod decorator so that we can reuse this later.
        d = []
        for col in df.columns:
            d.append(go.Scatter(x=df.index, y=df[col], name=col))
        fig = go.Figure(data=d)
        fig.update_layout(
            title=labels['title'],
            xaxis_title=labels['xaxis'],
            yaxis_title=labels['yaxis'],
            xaxis=dict(
                rangeselector=dict(
                    buttons=list(
                        [
                            dict(count=1, label="1m", step="month", stepmode="backward"),
                            dict(count=6, label="6m", step="month", stepmode="backward"),
                            dict(count=1, label="YTD", step="year", stepmode="todate"),
                            dict(count=1, label="1y", step="year", stepmode="backward"),
                            dict(step="all"),
                        ]
                    )
                ),
                rangeslider=dict(visible=True),
                type="date",
            ),
        )
        fig.show()

    def plot_profit_loss(self):
        # Plotting Profit/Loss (%)
        labels = {'title': '%s Profit/Loss (%%)' % self.ticker,
                  'xaxis': 'Date',
                  'yaxis': '%'}
        self.plot_this(pd.DataFrame(self.df["Profit/Loss (%)"]), labels)
        # p1.set_ylabel('%')
        # plt.show()

    def __repr__(self):
        return "Stock(%s)" % self.ticker


class Position(Stock):
    # A Position is a special kind of Stock: one we own
    # And it takes a list of movements as input
    def __init__(self, ticker, df):
        self.df = df
        self.ticker = ticker
        self.prices = self._load_prices(self.df.index[0])
        self._populate()
        self.current_value = round(self.df['Current Value'].iloc[-1], 2)
        self.invested_balance = round(self.df["Invested Balance"].iloc[-1], 2)
        self.profit = round(self.df['Profit/Loss'].iloc[-1], 2)
        self.profit_percentage = round(self.df['Profit/Loss (%)'].iloc[-1], 2)
        self.shares_owned = self.df["Owned"].iloc[-1]
        self.current_price = round(self.prices.iloc[-1], 2)
        # self.avg_purchase_price = self.df["Invested Balance"].iloc[-1]/self.df["Owned"].iloc[-1]

    def _populate(self):
        # calculate the num of shares currently owned
        self.df["Owned"] = self.df["Quantity"].cumsum()
        # calculate the invested balance
        self.df["Invested Balance"] = self.df["Purchase Price"] * self.df["Quantity"]
        self.df["Invested Balance"] = self.df["Invested Balance"].cumsum()
        self.df["Realized Profit/Loss"] = -(self.df["Quantity"] * self.df["Purchase Price"]). \
            where(self.df["Quantity"] < 0)
        self.df["Realized Profit/Loss"] = self.df["Realized Profit/Loss"].cumsum()
        # Making sure we have the same index as the historical series of stock prices
        self.df = self.df.reindex(self.prices.index)
        # Filling the reindexed dataframe with the right data #
        self.df["Quantity"].fillna(0, inplace=True)
        self.df["Purchase Price"].fillna(0, inplace=True)
        self.df["Realized Profit/Loss"].ffill(inplace=True)
        self.df["Realized Profit/Loss"].fillna(0, inplace=True)
        self.df.ffill(inplace=True)
        # --------------------------------------------------- #
        self.df['Current Value'] = self.df['Owned'] * self.prices
        self.df['Profit/Loss'] = self.df['Current Value'] - self.df["Invested Balance"]
        self.df['Profit/Loss (%)'] = ((self.df["Realized Profit/Loss"] + self.df['Current Value']) /
                                      (self.df["Realized Profit/Loss"] + self.df["Invested Balance"]) - 1) * 100
        # Reindexing over the full year with all the days to take into account for market closed days
        start_date = self.df.index.min()
        end_date = self.df.index.max()
        period = pd.date_range(start_date, end_date)
        # sometimes it seems that Yahoo Finance returns duplicated rows... deduplicating here
        self.df = self.df[~self.df.index.duplicated(keep='first')]
        self.df = self.df.reindex(period, method='ffill')

    def plot_value(self):
        # Plotting the money value against the invested balance
        labels = {'title': '%s Position Value' % self.ticker,
                  'xaxis': 'Date',
                  'yaxis': 'USD'}
        self.plot_this(self.df[['Current Value', 'Invested Balance', 'Profit/Loss']], labels)

    def monthly_report(self):
        # Prints end of month portfolio report
        end = self.df.index.max()
        today = self.df.index.min()
        while today < end - MonthEnd(1):
            today = today + MonthEnd(1)
            print(today.date(), "%.2f" % self.df.loc[today]["Profit/Loss (%)"])

    def __repr__(self):
        return "Position(%s)" % self.ticker


class Portfolio:
    def __init__(self, positions):
        self.positions = positions
        self.df = pd.DataFrame()
        self._populate()
        self.tickers = [self.positions[x].ticker for x in self.positions.keys()]
        self.current_value = round(self.df['Current Value'].iloc[-1], 2)
        self.allocation = self._get_positions_size()
        self.position_values = self._get_positions_values()

    def __repr__(self):
        return "Portfolio(%.2f USD)" % self.current_value

    def _populate(self):
        for p in self.positions.keys():
            if not self.df.empty:
                self.df = self.df.add(self.positions[p].df[["Invested Balance", "Current Value"]], fill_value=0)
            else:
                self.df = self.positions[p].df[["Invested Balance", "Current Value"]]

        # Calculating Profit/Loss of the whole portfolio day by day
        self.df['Profit/Loss'] = self.df['Current Value'] - self.df['Invested Balance']
        self.df['Profit/Loss (%)'] = (self.df['Current Value'] / self.df['Invested Balance'] - 1) * 100

    def plot_all_positions(self):
        # Calculating optimal matrix size
        size = len(self.positions.keys())
        rows = round(math.sqrt(size))
        cols = math.ceil(size / rows)
        # Making subplot matrix
        fig = make_subplots(rows=rows, cols=cols, start_cell="top-left",
                            subplot_titles=list(self.positions.keys()))
        c = 1
        for p in self.positions.keys():
            row = math.ceil(c / cols)
            col = cols if c % cols == 0 else c % cols
            pos = self.positions[p]
            fig.add_trace(go.Scatter(x=pos.df.index, y=pos.df['Profit/Loss (%)'], name=pos.ticker),
                          row=row, col=col)
            fig.update_yaxes(title_text="%", row=row, col=col)
            c += 1
        fig.update_layout(title_text="Profit/Loss (%) of individual positions")
        fig.show()

    def plot_profit_loss(self, benchmark="^NDX"):
        # Plotting Profit/Loss (%)
        p1 = self.df["Profit/Loss (%)"]
        index = self.load_benchmark(benchmark)
        index._rebase_profits_from(p1.index.min())
        idx = index.df["Profit/Loss (%)"]
        t = pd.concat((p1, idx.rename(index.ticker)), axis=1)
        labels = {'title': 'Portfolio Profit/Loss (%)',
                  'xaxis': 'Date',
                  'yaxis': '%'}
        Stock.plot_this(t, labels)

    def plot_value(self):
        # Plotting the money value against the invested balance
        labels = {'title': 'Portfolio Value',
                  'xaxis': 'Date',
                  'yaxis': 'USD'}
        Stock.plot_this(self.df[['Current Value', 'Invested Balance', 'Profit/Loss']], labels)

    def load_benchmark(self, index):
        # Loads a benchmark position
        return Stock(index, self.df.index.min())

    def rebalance(self, new):
        # TODO: This takes a new intended allocation and gives the best option to reach it
        # new is a list of tuples, exactly like self.allocation
        pass

    def _get_positions_size(self):
        sizes = []
        for p in self.positions.keys():
            share = round(self.positions[p].current_value / self.current_value * 100, 2)
            sizes.append((self.positions[p].ticker, share))
            # We create a property of the positions itself to remember its size
            # Note: The position itself can't have a size without being associated to a Portfolio
            self.positions[p].size = share
        sizes.sort(key=lambda x: x[1], reverse=True)
        return sizes

    def _get_positions_values(self):
        values = []
        for p in self.positions.keys():
            share = round(self.positions[p].current_value / self.current_value * 100, 2)
            values.append((self.positions[p].ticker, self.positions[p].current_value))
        values.sort(key=lambda x: x[1], reverse=True)
        return values

    def monthly_report(self):
        # Prints end of month portfolio report
        end = self.df.index.max()
        today = self.df.index.min()
        print("\nMonthly Report\n____________________")
        while today < end - MonthEnd(1):
            today = today + MonthEnd(1)
            print(today.date(), "%.2f" % self.df.loc[today]["Profit/Loss (%)"])


class Cache:
    # The intention of the cache is more to preserve old data than to save on API calls
    def __init__(self, cache_file):
        self.found = False
        self.cache_dir = "./cache"
        if not os.path.exists(self.cache_dir):
            os.mkdir(self.cache_dir)
        self.cache_file = os.path.join(self.cache_dir, cache_file)

    def save(self, data):
        self.data = data
        return pd.to_pickle(self.data, self.cache_file)

    def load(self):
        if os.path.exists(self.cache_file):
            self.data = pd.read_pickle(self.cache_file)
            self.found = True
        else:
            self.data = pd.Series([], dtype='float64')
        return self.data

    def end_date(self):
        if self.data.empty:
            self.data = self.load()
        return self.data.index[-1]

    def start_date(self):
        if self.data.empty:
            self.data = self.load()
        return self.data.index[0]


def main(infile):
    print("loading %s" % infile)
    positions = YahooCsv(infile).load()
    portfolio = Portfolio(positions)
    portfolio.plot_profit_loss("^IXIC")
    portfolio.monthly_report()
    portfolio.plot_all_positions()
    portfolio.plot_value()
    print('Allocations')
    for x in portfolio.allocation:
        print(x[0],', ', x[1])
    print('Position values')
    for x in portfolio.position_values:
        print(x[0], ', ', x[1])


def parse_arguments():
    parser = argparse.ArgumentParser(description='''Loads a CSV containing the export of your Yahoo Finance portfolio
    [get it here for instance: https://finance.yahoo.com/portfolio/p_0/view] and plots profit/loss graphs for it''')

    parser.add_argument('input_file', metavar='FILEPATH', type=str,
                        help='The CSV of the exported portfolio')

    arguments = parser.parse_args()
    return arguments


if __name__ == "__main__":
    args = parse_arguments()
    infile = args.input_file
    main(infile)
