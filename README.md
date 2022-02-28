# TinyPortfolioTracker

## Legal note

Yahoo!, Y!Finance, and Yahoo! finance are registered trademarks of Yahoo, Inc.

tinyportfoliotracker is not affiliated, endorsed, or vetted by Yahoo, Inc. It's an open-source tool that uses Yahoo's publicly available APIs, and is intended for research and educational purposes.

You should refer to Yahoo!'s terms of use for details on your rights to use the actual data downloaded.

## Note(2)
None of the graphs here are coming from real investments, I just made up some test data in a bogus portfolio on Yahoo!Finance ;)

## Introduction
TinyPortfolioTracker is a small set of libraries that lets you track your portfolio performance by simply exporting your portfolio that you find [here](https://finance.yahoo.com/portfolio/p_0/view) by clicking on the top right corner "Export" button.

![](images/export.png)

It retrieves Yahoo!Finance data using `pandas-datareader` and tries to keep things small and simple offering a pythonic interface to interacting with your portfolio.

## Usage

Import the needed classes from `yportfolio` module and load your positions from the CSV file you exported:
```python
from yportfolio import YahooCsv, Position, Portfolio

positions = YahooCsv("your_quotes.csv").load()
portfolio = Portfolio(positions)
```
### Positions
You will have `positions` which is a simple list containing all your positions, with their historical prices.
```python
In [6]: positions
Out[6]: {'AAPL': Position(AAPL), 'GOOG': Position(GOOG), 'MSFT': Position(MSFT)}

```
You can interact with each of these, discovering their current price, how much you have invested in them and how many shares you own. Most importantly you can plot the Profit/Loss (`position.plot_profit_loss()`)
![](images/aapl_profit.png)

...and their Value graph (`position.plot_value()`).
![](images/aapl.png)

### Portfolio
You will also have access to the aggregate portfolio value (`portfolio.current_value`) and its allocation among the different positions (`portfolio.allocation`). And as you do with each position you will have a chance to plot the Profit/Loss (`portfolio.plot_profit_loss()`) 

![](images/plot_profit.png)
    
and its Value graph (`portfolio.plot_value()`).

![](images/plot_value.png)

and the combined profit/loss graph of all positions (`portfolio.plot_all_positions()`)

![](images/multiplot.png)

Lastly also a handy function to calculate a monthly report of your portfolio:

```python
In [7]: portfolio.monthly_report()
2021-08-31 2.37
2021-09-30 -5.70
2021-10-31 7.52
2021-11-30 8.65
2021-12-31 12.19
2022-01-31 6.24
```


# Caveats
 * This was a quick exercise and besides a bit of caching I have not added anything fancy
 * Logging is non existent
 * Testing has not been done, but if you end up using it and find bugs please report them here ;)



