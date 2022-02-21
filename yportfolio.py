import pandas as pd
import pandas_datareader as web
import datetime as dt
import numpy as np
import matplotlib.pyplot as plt
import os


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
        positions = []
        for t in tickers:
            p = movements.xs(t)
            p["Symbol"] = t
            positions.append(Position(t, p))

        return positions


class Position:
    def __init__(self, ticker, df):
        self.df = df
        self.ticker = ticker
        self.prices = self._load_prices()
        self._populate()
        self.current_value = self.df['Current Value'].iloc[-1]
        self.invested_balance = self.df["Invested Balance"].iloc[-1]
        self.profit = self.df['Profit/Loss'].iloc[-1]
        self.profit_percentage = self.df['Profit/Loss (%)'].iloc[-1]
        self.shares_owned = self.df["Owned"].iloc[-1]
        self.current_price = self.prices.iloc[-1]
        #self.avg_purchase_price = self.df["Invested Balance"].iloc[-1]/self.df["Owned"].iloc[-1]

    def _load_prices(self):
        # Check first if we already have some historical prices saved in cache
        cache = Cache(self.ticker + ".pkl")
        cached_prices = cache.load()
        if cache.found:
            print('cache found')
            start_date = cache.end_date() + dt.timedelta(days=1)
        else:
            print('cache not found')
            start_date = self.df.index[0]
        new_data = False
        if dt.datetime.today() > start_date:
            # Let's avoid making calls to Yahoo if not needed
            new_prices = web.DataReader(self.ticker, "yahoo", start=start_date)["Adj Close"]
            new_data = True
        if cache.found and new_data:
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
        # calculate the num of shares currently owned
        self.df["Owned"] = self.df["Quantity"].cumsum()
        # calculate the invested balance
        self.df["Invested Balance"] = self.df["Purchase Price"] * self.df["Quantity"]
        self.df["Invested Balance"] = self.df["Invested Balance"].cumsum()
        # Making sure we have the same index as the historical series of stock prices
        self.df = self.df.reindex(self.prices.index)
        # Filling the reindexed dataframe with the right data #
        self.df["Quantity"].fillna(0, inplace=True)
        self.df["Purchase Price"].fillna(0, inplace=True)
        self.df.ffill(inplace=True)
        # --------------------------------------------------- #
        print(self.df['Owned'].shape)
        self.df['Current Value'] = self.df['Owned'] * self.prices
        self.df['Profit/Loss'] = self.df['Current Value'] - self.df["Invested Balance"]
        self.df['Profit/Loss (%)'] = (self.df['Current Value']/self.df["Invested Balance"] - 1) * 100
        # Reindexing over the full year with all the days to take into account for market closed days
        start_date = self.df.index.min()
        end_date = self.df.index.max()
        period = pd.date_range(start_date, end_date)
        self.df  = self.df.reindex(period, method='ffill')



    def plot_profit_loss(self):
        # Plotting Profit/Loss (%)
        p1 = self.df["Profit/Loss (%)"].plot(grid=True, legend=True, title="%s Position Profit/Loss (%%)" % self.ticker)
        p1.set_ylabel('%')
        plt.show()

    def plot_value(self):
        # Plotting the money value against the invested balance
        p2 = self.df[['Current Value', 'Invested Balance', 'Profit/Loss']].plot(grid=True, title="%s Position Value" % self.ticker)
        p2.set_ylabel("USD")
        plt.show()

    def __repr__(self):
        return "Position(%s)" % self.ticker

        
class Portfolio:
    def __init__(self, positions):
        self.positions = positions
        self.df = pd.DataFrame()
        self._populate()
        
    def _populate(self):
        for p in self.positions:
            if not self.df.empty:
                self.df = self.df.add(p.df[["Invested Balance", "Current Value"]], fill_value=0)
            else:
                self.df = p.df[["Invested Balance", "Current Value"]]

        # Calculating Profit/Loss of the whole portfolio day by day
        self.df['Profit/Loss'] = self.df['Current Value'] - self.df['Invested Balance']
        self.df['Profit/Loss (%)'] = (self.df['Current Value'] / self.df['Invested Balance'] - 1) * 100

    def plot_profit_loss(self):
        # Plotting Profit/Loss (%)
        p1 = self.df["Profit/Loss (%)"].plot(grid=True, legend=True, title="Portfolio Profit/Loss (%)")
        p1.set_ylabel('%')
        plt.show()

    def plot_value(self):
        # Plotting the money value against the invested balance
        p2 = self.df[['Current Value', 'Invested Balance', 'Profit/Loss']].plot(grid=True, title="Portfolio Value")
        p2.set_ylabel("USD")
        plt.show()


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
            self.data = pd.Series([],dtype='float64')
        return self.data

    def end_date(self):
        if self.data.empty:
            self.data = self.retrieve()
        return self.data.index[-1]

    def start_date(self):
        if self.data.empty:
            self.data = self.retrieve()
        return self.data.index[0]



def main():
    positions = YahooCsv("/Users/stemosco/Downloads/quotes_2.csv").load()
    portfolio = Portfolio(positions)
    portfolio.plot_value()

