## StockToy: Stock Analysis Dashboard

Stock toy is a stock market peformance dash board that tracks key metrics for a stock compairing the stock against the S&P 500. Giving information for price returns and dividends,


Features

## Main Analysis Tab
- **Price and Volatility Chart**: Track historical closing prices and calculate annualized historical volatility (realized not implied)
- **Beta Analysis**: Calculate and visualize beta coefficient against the S&P 500 (SPY) with regression statistics
- **Normalized Price Comparison**: Compare stock performance against SPY with normalized pricing
- **Return Distribution Analysis**: View histograms, statistics, and violin plots of return distributions

### Dividend Tab
- **Dividend History**: Track dividend payments over time
- **Dividend Yield Analysis**: Calculate and visualize dividend yields with price correlation
- **Dividend Growth**: View statistics on dividend growth rates and yields

## Installation

### Prerequisites
- Python 3.7+
- Tkinter (usually included with Python)

### Dependencies
Install required packages:

```bash
pip install matplotlib seaborn numpy pandas yfinance statsmodels scipy
```

### Running the Application
```bash
python StockToy.py
```

## Usage

1. Enter a stock ticker symbol (default: KO)
2. Specify the number of years for analysis (default: 5)
3. Click "Analyze" to generate visualizations
4. Switch between tabs to view different analysis types

## Technical Details

The application is built with:
- **Tkinter**: For the user interface
- **Matplotlib & Seaborn**: For data visualization
- **yfinance**: For retrieving stock data from Yahoo Finance
- **pandas & numpy**: For data manipulation
- **statsmodels**: For statistical analysis

## Code Structure

- `StockAnalysisApp`: Main application class
  - `create_main_analysis_tab`: Creates the main analysis UI
  - `create_dividend_tab`: Creates the dividend analysis UI
  - `analyze_stock`: Performs the main stock analysis
  - `analyze_dividends`: Performs dividend analysis
  - Various `update_*_plot` methods: Generate individual visualizations

## Contributing

1. Go nuts

## License

No clue lol


## Acknowledgments

- [Yahoo Finance](https://finance.yahoo.com/) for providing the financial data API
- [matplotlib](https://matplotlib.org/) and [seaborn](https://seaborn.pydata.org/) for visualization capabilities
