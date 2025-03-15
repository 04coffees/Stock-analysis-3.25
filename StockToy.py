import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import yfinance as yf
import datetime as dt
import seaborn as sns
import numpy as np
import pandas as pd
import statsmodels.api as sm
import scipy.stats as stats
import matplotlib.gridspec as gridspec
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter

class StockAnalysisApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Stock Analysis Dashboard")
        self.root.geometry("1200x800")
        
        # Create notebook for tabs
        self.notebook = ttk.Notebook(root)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # Create main frame
        main_frame = ttk.Frame(self.notebook)
        self.notebook.add(main_frame, text="Main Analysis")
        
        # Create dividend frame
        dividend_frame = ttk.Frame(self.notebook)
        self.notebook.add(dividend_frame, text="Dividend")
        
        # Main Analysis Tab
        self.create_main_analysis_tab(main_frame)
        
        # Dividend Tab
        self.create_dividend_tab(dividend_frame)
        
        # Run initial analysis
        self.analyze_stock()
    
    def create_control_frame(self, frame, tab_name):
        """Create a control frame with stock selection controls"""
        control_frame = ttk.Frame(frame)
        control_frame.pack(fill=tk.X, pady=5)
        
        # Stock ticker entry
        ttk.Label(control_frame, text="Stock Ticker:").pack(side=tk.LEFT, padx=5)
        ticker_entry = ttk.Entry(control_frame, width=10)
        ticker_entry.pack(side=tk.LEFT, padx=5)
        ticker_entry.insert(0, "KO")  # Default to KO
        
        # Years selection
        ttk.Label(control_frame, text="Years:").pack(side=tk.LEFT, padx=5)
        years_entry = ttk.Entry(control_frame, width=5)
        years_entry.pack(side=tk.LEFT, padx=5)
        years_entry.insert(0, "5")  # Default to 5 years
        
        # Analyze button
        analyze_button = ttk.Button(control_frame, text="Analyze", 
                                  command=lambda: self.analyze_stock(tab_name))
        analyze_button.pack(side=tk.LEFT, padx=20)
        
        # Status label
        status_label = ttk.Label(control_frame, text="")
        status_label.pack(side=tk.LEFT, padx=5)
        
        # Store the controls in the class instance
        if tab_name == "main":
            self.main_ticker_entry = ticker_entry
            self.main_years_entry = years_entry
            self.main_status_label = status_label
        else:
            self.dividend_ticker_entry = ticker_entry
            self.dividend_years_entry = years_entry
            self.dividend_status_label = status_label
        
        return control_frame

    def create_main_analysis_tab(self, frame):
        # Create control frame
        self.create_control_frame(frame, "main")
        
        # Create plots frame
        self.plots_frame = ttk.Frame(frame)
        self.plots_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create frames for individual plots
        self.price_volatility_frame = ttk.LabelFrame(self.plots_frame, text="Price and Volatility Chart")
        self.price_volatility_frame.grid(row=0, column=0, padx=5, pady=5, sticky="nsew")
        
        self.beta_frame = ttk.LabelFrame(self.plots_frame, text="Beta Analysis")
        self.beta_frame.grid(row=1, column=0, padx=5, pady=5, sticky="nsew")
        
        self.normalized_price_frame = ttk.LabelFrame(self.plots_frame, text="Normalized Close Prices")
        self.normalized_price_frame.grid(row=1, column=1, padx=5, pady=5, sticky="nsew")
        
        self.histogram_frame = ttk.LabelFrame(self.plots_frame, text="Histogram and Statistics")
        self.histogram_frame.grid(row=0, column=1, padx=5, pady=5, sticky="nsew")
        
        # Configure grid weights
        self.plots_frame.grid_columnconfigure(0, weight=1)
        self.plots_frame.grid_columnconfigure(1, weight=1)
        self.plots_frame.grid_rowconfigure(0, weight=1)
        self.plots_frame.grid_rowconfigure(1, weight=1)
    
    def create_dividend_tab(self, frame):
        # Create control frame
        self.create_control_frame(frame, "dividend")
        
        # Create a container frame for the plots
        plots_container = ttk.Frame(frame)
        plots_container.pack(fill=tk.BOTH, expand=True)
        
        # Create left side frame for plots
        left_frame = ttk.Frame(plots_container)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Create right side frame for text
        right_frame = ttk.Frame(plots_container)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Create frames for individual plots
        self.dividend_plot1_frame = ttk.LabelFrame(left_frame, text="Dividend History")
        self.dividend_plot1_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.dividend_plot2_frame = ttk.LabelFrame(left_frame, text="Dividend Yield Analysis")
        self.dividend_plot2_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.dividend_text_frame = ttk.LabelFrame(right_frame, text="Dividend Information")
        self.dividend_text_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
    
    def analyze_stock(self, tab_name="main"):
        """Perform stock analysis and update all plots"""
        try:
            # Get the appropriate status label and controls based on the tab
            if tab_name == "main":
                status_label = self.main_status_label
                ticker_entry = self.main_ticker_entry
                years_entry = self.main_years_entry
            else:
                status_label = self.dividend_status_label
                ticker_entry = self.dividend_ticker_entry
                years_entry = self.dividend_years_entry
            
            status_label.config(text="Analyzing... Please wait.")
            self.root.update()
            
            # Get inputs
            stock = ticker_entry.get().strip().upper()
            try:
                years = float(years_entry.get())
            except ValueError:
                status_label.config(text="Error: Years must be a number")
                return
            
            if tab_name == "main":
                # Get data from yfinance
                end_date = dt.date.today()
                start_date = end_date - dt.timedelta(days=years*365)
                Assets = [stock, 'SPY']
                
                df = yf.download(Assets, start=start_date, end=end_date)
                
                # Process data
                df_reset = df['Close'][[stock, 'SPY']].reset_index()
                
                df_reset[stock + ' Return'] = (df_reset[stock] - df_reset[stock].shift(1)) / df_reset[stock].shift(1)
                df_reset['SPY Return'] = (df_reset['SPY'] - df_reset['SPY'].shift(1)) / df_reset['SPY'].shift(1)
                df_reset['Log Return'] = np.log(df_reset[stock] / df_reset[stock].shift(1))
                df_reset['HV (Daily)'] = df_reset['Log Return'].rolling(window=20).std()
                df_reset['HV (Annual)'] = df_reset['HV (Daily)'] * np.sqrt(252)
                
                df_reset.rename(columns={stock: stock + ' Close', 'SPY': 'SPY Close'}, inplace=True)
                
                new_df = df_reset[['Date', stock + ' Close', 'SPY Close', 'SPY Return', 
                                  stock + ' Return', 'Log Return', 'HV (Daily)', 'HV (Annual)']]
                
                new_df['Date'] = pd.to_datetime(new_df['Date'])
                new_df = new_df.dropna(subset=[stock + ' Close', 'HV (Annual)'])
                
                # Update all plots
                self.update_price_volatility_plot(new_df, stock)
                self.update_beta_plot(new_df, stock)
                self.update_normalized_price_plot(new_df, stock)
                self.update_histogram_plot(new_df, stock)
            
            # Update dividend analysis
            self.analyze_dividends()
            
            status_label.config(text=f"Analysis complete for {stock}")
            
        except Exception as e:
            status_label.config(text=f"Error: {str(e)}")
    
    def update_price_volatility_plot(self, data, stock):
        """Update the price and volatility plot"""
        # Clear previous plot
        for widget in self.price_volatility_frame.winfo_children():
            widget.destroy()
            
        # Create figure and subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 9), gridspec_kw={'height_ratios': [2, 1]})
        
        # Plot close price
        sns.lineplot(data=data, x='Date', y=stock + ' Close', ax=ax1)
        ax1.set_title(stock.upper() + ' Close Price')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Close Price')
        
        # Plot historical volatility
        sns.lineplot(data=data, x='Date', y='HV (Annual)', ax=ax2)
        ax2.set_title(stock.upper() + ' HV (Annual)')
        ax2.set_xlabel('Date')
        ax2.set_ylabel('HV (Annual)')
        
        # Calculate and plot median line for HV (Annual)
        median_hv = data['HV (Annual)'].median()
        ax2.axhline(median_hv, color='red', linestyle='--', label=f'Median HV (Annual): {median_hv:.4f}')
        ax2.legend()
        
        # Rotate x-axis labels for readability
        for ax in [ax1, ax2]:
            for label in ax.get_xticklabels():
                label.set_rotation(45)
        
        fig.tight_layout()
        
        # Embed in tkinter
        canvas = FigureCanvasTkAgg(fig, master=self.price_volatility_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def update_beta_plot(self, data, stock):
        """Update the beta analysis plot"""
        # Clear previous plot
        for widget in self.beta_frame.winfo_children():
            widget.destroy()
            
        # Prepare the data for regression
        X = data['SPY Return'].dropna()
        y = data[stock + ' Return'].dropna()
        
        # Align the data
        X, y = X.align(y, join='inner')
        
        # Add constant for regression
        X_const = sm.add_constant(X)
        
        # Perform regression
        model = sm.OLS(y, X_const).fit()
        
        # Get regression line
        regression_line = model.params[0] + model.params[1] * X
        
        # Calculate statistics
        covariance = np.cov(X, y)[0, 1] * 100
        variance_spy = np.var(X) * 100
        beta = model.params[1]
        intercept = model.params[0]
        r_squared = model.rsquared
        
        # Create figure and layout
        fig = Figure(figsize=(10, 4), dpi=100)
        
        # Create scatter plot on the left
        ax1 = fig.add_subplot(121)
        ax1.scatter(X, y, alpha=0.6)
        ax1.plot(X, regression_line, color='red', linestyle='--')
        ax1.axhline(0, color='black', linewidth=0.5)
        ax1.axvline(0, color='black', linewidth=0.5)
        ax1.set_title(f'SPY Returns vs. {stock} Returns')
        ax1.set_xlabel('SPY Returns')
        ax1.set_ylabel(f'{stock} Returns')
        
    
        
        
        # Create text area for stats on the right
        ax2 = fig.add_subplot(122)
        ax2.axis('off')
        
        # Display stats
        stats_text = (f'Beta: {beta:.4f}\n'
                      f'Y-Intercept: {intercept:.6f}\n'
                      f'R-Squared: {r_squared:.4f}\n'
                      f'Covariance: {covariance:.4f}%\n'
                      f'SPY Variance: {variance_spy:.4f}%\n')
                      
        ax2.text(0.05, 0.5, stats_text, fontsize=12, verticalalignment='center')
        
        fig.tight_layout()
        
        # Embed in tkinter
        canvas = FigureCanvasTkAgg(fig, master=self.beta_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def update_normalized_price_plot(self, data, stock):
        """Update the normalized price plot"""
        # Clear previous plot
        for widget in self.normalized_price_frame.winfo_children():
            widget.destroy()
        
        # Create a container frame for the plot and text
        container = ttk.Frame(self.normalized_price_frame)
        container.pack(fill=tk.BOTH, expand=True)
        
        # Create plot frame (wider)
        plot_frame = ttk.Frame(container)
        plot_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        
        # Create text frame (narrower)
        text_frame = ttk.Frame(container)
        text_frame.pack(side=tk.RIGHT, fill=tk.Y, expand=False, padx=(5, 0))
        
        # Normalize the close prices so that both start at 1
        data[stock + ' Close Normalized'] = data[stock + ' Close'] / data[stock + ' Close'].iloc[0]
        data['SPY Close Normalized'] = data['SPY Close'] / data['SPY Close'].iloc[0]
        
        # Calculate returns
        stock_total_return = (data[stock + ' Close'].iloc[-1] / data[stock + ' Close'].iloc[0] - 1) * 100
        spy_total_return = (data['SPY Close'].iloc[-1] / data['SPY Close'].iloc[0] - 1) * 100
        
        # Calculate annualized returns
        years = (data['Date'].iloc[-1] - data['Date'].iloc[0]).days / 365.25
        stock_annual_return = ((1 + stock_total_return/100) ** (1/years) - 1) * 100
        spy_annual_return = ((1 + spy_total_return/100) ** (1/years) - 1) * 100
        
        # Create figure (wider)
        fig = Figure(figsize=(7, 3), dpi=100)
        ax = fig.add_subplot(111)
        
        # Plot normalized close prices
        sns.lineplot(data=data, x='Date', y=stock + ' Close Normalized', ax=ax, label=stock + ' Close Normalized')
        sns.lineplot(data=data, x='Date', y='SPY Close Normalized', ax=ax, label='SPY Close Normalized')
        
        # Set title and labels
        ax.set_title('Normalized Close Prices for ' + stock.upper() + ' and SPY')
        ax.set_xlabel('Date')
        ax.set_ylabel('Normalized Close Price')
        
        # Rotate x-axis labels for readability
        for label in ax.get_xticklabels():
            label.set_rotation(45)
        
        fig.tight_layout()
        
        # Embed plot in tkinter
        canvas = FigureCanvasTkAgg(fig, master=plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Create text widget for returns (narrower)
        text_widget = tk.Text(text_frame, wrap=tk.WORD, padx=5, pady=10, width=25, height=10)
        text_widget.pack(fill=tk.Y, expand=False)
        
        # Define colors
        stock_color = "green" if stock_total_return >= 0 else "red"
        spy_color = "green" if spy_total_return >= 0 else "red"
        
        # Format and insert the information with color coding
        text_widget.tag_configure(stock_color, foreground=stock_color)
        text_widget.tag_configure(spy_color, foreground=spy_color)
        
        # Insert text with color tags
        text_widget.insert(tk.END, "Returns Analysis:\n----------------\n")
        text_widget.insert(tk.END, f"{stock}:\n")
        text_widget.insert(tk.END, "Total Return: ", stock_color)
        text_widget.insert(tk.END, f"{stock_total_return:.2f}%\n", stock_color)
        text_widget.insert(tk.END, "Annualized: ", stock_color)
        text_widget.insert(tk.END, f"{stock_annual_return:.2f}%\n\n", stock_color)
        
        text_widget.insert(tk.END, "SPY:\n")
        text_widget.insert(tk.END, "Total Return: ", spy_color)
        text_widget.insert(tk.END, f"{spy_total_return:.2f}%\n", spy_color)
        text_widget.insert(tk.END, "Annualized: ", spy_color)
        text_widget.insert(tk.END, f"{spy_annual_return:.2f}%\n", spy_color)
        
        text_widget.config(state=tk.DISABLED)  # Make text read-only
    
    def update_histogram_plot(self, data, stock):
        """Update the histogram and statistics plot"""
        # Clear previous plot
        for widget in self.histogram_frame.winfo_children():
            widget.destroy()
        
        # Create figure with 2x2 grid layout
        fig = Figure(figsize=(12, 9), dpi=100)
        gs = gridspec.GridSpec(2, 2, figure=fig, width_ratios=[2, 1], height_ratios=[2, 1])
        
        # Left side plots (histogram and stats)
        ax1 = fig.add_subplot(gs[0, 0])  # Histogram
        ax2 = fig.add_subplot(gs[1, 0])  # Stats
        
        # Right side plots (violin plot and regression results)
        ax3 = fig.add_subplot(gs[0, 1])  # Violin plot
        ax4 = fig.add_subplot(gs[1, 1])  # Regression results
        
        # Plot histogram of stock returns
        sns.histplot(data[stock + ' Return'].dropna(), bins=50, kde=True, ax=ax1)
        ax1.set_title(f'Histogram of {stock} Returns')
        ax1.set_xlabel('Returns')
        ax1.set_ylabel('Frequency')
        
        # Create violin plot comparing stock and SPY returns
        returns_data = pd.DataFrame({
            stock: data[stock + ' Return'].dropna(),
            'SPY': data['SPY Return'].dropna()
        })
        
        # Melt the data for violin plot
        returns_melted = returns_data.melt().assign(x='Returns')
        
        # Create split violin plot with inner quartiles
        sns.violinplot(data=returns_melted, x='x', y='value', 
                      hue='variable', split=True, inner='quart', ax=ax3)
        ax3.set_title('Return Distribution Comparison')
        ax3.set_xlabel('')
        ax3.set_ylabel('Returns')
        
        # Calculate statistics for both stock and SPY
        stock_mean = data[stock + ' Return'].mean()
        stock_std = data[stock + ' Return'].std()
        spy_mean = data['SPY Return'].mean()
        spy_std = data['SPY Return'].std()
        
        # Add text box with statistics in top right corner
        stats_text = f"{stock}:\nMean: {stock_mean:.4f}\nStd: {stock_std:.4f}\n\nSPY:\nMean: {spy_mean:.4f}\nStd: {spy_std:.4f}"
        ax3.text(1.5, ax3.get_ylim()[1], stats_text, fontsize=10, 
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'),
                verticalalignment='top')
        
        # Calculate basic statistics
        returns = data[stock + ' Return'].dropna()
        mean_return = np.mean(returns)
        median_return = np.median(returns)
        std_deviation = np.std(returns)
        min_return = np.min(returns)
        max_return = np.max(returns)
        skewness = stats.skew(returns)
        kurtosis = stats.kurtosis(returns)
        q25, q50, q75 = np.percentile(returns, [25, 50, 75])
        shapiro_test = stats.shapiro(returns)
        
        # Display formal summary
        summary_text = (f"Formal Summary of Histogram for {stock} Returns:\n"
                        f"---------------------------------------------------\n"
                        f"Mean Return:             {mean_return:.4f}\n"
                        f"Median Return:           {median_return:.4f}\n"
                        f"Standard Deviation:      {std_deviation:.4f}\n"
                        f"Minimum Return:          {min_return:.4f}\n"
                        f"Maximum Return:          {max_return:.4f}\n"
                        f"Skewness:                {skewness:.4f}\n"
                        f"Kurtosis:                {kurtosis:.4f}\n"
                        f"25th Percentile (Q1):    {q25:.4f}\n"
                        f"50th Percentile (Median):{q50:.4f}\n"
                        f"75th Percentile (Q3):    {q75:.4f}\n"
                        f"Shapiro-Wilk p-value:    {shapiro_test.pvalue:.4f} ")
        
        # Prepare the data for regression
        X = data['SPY Return'].dropna()
        y = data[stock + ' Return'].dropna()
        
        # Align the data
        X, y = X.align(y, join='inner')
        
        # Add constant for regression
        X_const = sm.add_constant(X)
        
        # Perform regression
        model = sm.OLS(y, X_const).fit()
        
        # Create text area for stats on the bottom
        ax2.axis('off')
        ax4.axis('off')
        
        # Display formal summary in bottom left
        ax2.text(0.05, 0.5, summary_text, fontsize=8, verticalalignment='center', family='monospace')
        
        # Display regression results on the bottom right
        regression_summary_text = model.summary().as_text()
        ax4.text(0.05, 0.5, regression_summary_text, fontsize=8, verticalalignment='center', family='monospace')
        
        fig.tight_layout()
        
        # Embed in tkinter
        canvas = FigureCanvasTkAgg(fig, master=self.histogram_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def analyze_dividends(self):
        """Perform dividend analysis and update all plots"""
        try:
            # Get inputs from dividend tab controls
            stock = self.dividend_ticker_entry.get().strip().upper()
            try:
                years = float(self.dividend_years_entry.get())
            except ValueError:
                self.dividend_status_label.config(text="Error: Years must be a number")
                return
            
            # Get today's date - make it timezone-aware to match yfinance data
            end_date = pd.Timestamp.today().tz_localize('UTC')
            start_date = end_date - pd.Timedelta(days=years*365)
            
            # Download the dividend data
            ticker = yf.Ticker(stock)
            dividends = ticker.dividends
            dividends = dividends[(dividends.index >= start_date) & (dividends.index <= end_date)]
            
            # Download the historical stock price data
            df = ticker.history(start=start_date, end=end_date)
            
            # Separate dividends and daily stock closing prices into different tables
            dividends_table = dividends.to_frame(name='Dividends')
            daily_close_table = df[['Close']].rename(columns={'Close': 'Daily Close'})
            
            # Order both tables from newest to oldest (for display purposes)
            dividends_table_display = dividends_table.sort_index(ascending=False)
            daily_close_table_display = daily_close_table.sort_index(ascending=False)
            
            # For calculations, we need chronological order (oldest to newest)
            dividends_table_calc = dividends_table.sort_index(ascending=True)
            dividend_dates = dividends_table_calc.index.tolist()
            
            # Create a list to store average closing prices
            average_closes = []
            
            # For each dividend period
            for i in range(len(dividend_dates) - 1):
                period_start = dividend_dates[i]
                period_end = dividend_dates[i + 1]
                period_prices = daily_close_table.loc[period_start:period_end]
                
                if not period_prices.empty:
                    avg_price = period_prices['Daily Close'].mean()
                    average_closes.append((period_end, avg_price))
            
            # Handle the last dividend period to current date
            if len(dividend_dates) > 0:
                last_div_date = dividend_dates[-1]
                last_period_prices = daily_close_table.loc[last_div_date:]
                
                if not last_period_prices.empty:
                    last_avg_price = last_period_prices['Daily Close'].mean()
                    last_date = daily_close_table.index[-1]
                    average_closes.append((last_date, last_avg_price))
            
            # Create DataFrame from average closes
            avg_close_df = pd.DataFrame(average_closes, columns=['Date', 'Average Close'])
            avg_close_df.set_index('Date', inplace=True)
            avg_close_df = avg_close_df.sort_index(ascending=False)
            
            # Merge the dividends table with the average closes table
            consolidated_table = dividends_table_display.merge(avg_close_df, left_index=True, right_index=True, how='left')
            
            # Update all dividend plots
            self.update_dividend_history_plot(consolidated_table, stock, years)
            self.update_dividend_yield_plot(consolidated_table, stock, years)
            self.update_dividend_info(consolidated_table, daily_close_table, dividends_table, stock)
            
        except Exception as e:
            self.dividend_status_label.config(text=f"Error in dividend analysis: {str(e)}")

    def update_dividend_history_plot(self, consolidated_table, stock, years):
        """Update the dividend history plot"""
        # Clear previous plot
        for widget in self.dividend_plot1_frame.winfo_children():
            widget.destroy()
            
        # Create figure
        fig = Figure(figsize=(10, 4), dpi=100)
        ax = fig.add_subplot(111)
        
        # Plot dividend data
        dividends = consolidated_table['Dividends'].sort_index(ascending=False)
        ax.plot(dividends.index, dividends.values, linestyle='-')
        ax.set_title(f'{stock} Quarterly Dividend past {years} years')
        ax.set_xlabel('Date')
        ax.set_ylabel('Dividend Amount ($)')
        ax.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f'${x:.2f}'))
        ax.grid(True)
        
        # Rotate x-axis labels for readability
        for label in ax.get_xticklabels():
            label.set_rotation(45)
        
        fig.tight_layout()
        
        # Embed in tkinter
        canvas = FigureCanvasTkAgg(fig, master=self.dividend_plot1_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def update_dividend_yield_plot(self, consolidated_table, stock, years):
        """Update the dividend yield analysis plot"""
        # Clear previous plot
        for widget in self.dividend_plot2_frame.winfo_children():
            widget.destroy()
            
        # Create figure and subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 9), gridspec_kw={'height_ratios': [2, 1]})
        
        # Get data
        dividends = consolidated_table['Dividends'].sort_index(ascending=False)
        average_close = consolidated_table['Average Close'].sort_index(ascending=False)
        
        # Calculate annual dividend yield
        dividend_yield = (dividends * 4 / average_close) * 100
        
        # Plot dividend yield
        ax1.plot(dividend_yield.index, dividend_yield.values, linestyle='-')
        ax1.set_title(f'{stock} Annual Dividend Yield past {years} years')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Annual Dividend Yield (%)')
        ax1.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f'{x:.2f}%'))
        ax1.grid(True)
        
        # Plot close prices
        ax2.plot(average_close.index, average_close.values, linestyle='-')
        ax2.set_title(f'{stock} Close Prices past {years} years')
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Close Price ($)')
        ax2.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f'${x:.2f}'))
        ax2.grid(True)
        
        # Rotate x-axis labels for readability
        for ax in [ax1, ax2]:
            for label in ax.get_xticklabels():
                label.set_rotation(45)
        
        fig.tight_layout()
        
        # Embed in tkinter
        canvas = FigureCanvasTkAgg(fig, master=self.dividend_plot2_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def update_dividend_info(self, consolidated_table, daily_close_table, dividends_table, stock):
        """Update the dividend information text"""
        # Clear previous text
        for widget in self.dividend_text_frame.winfo_children():
            widget.destroy()
            
        # Get the newest and oldest data
        newest_close = daily_close_table.sort_index(ascending=False)['Daily Close'].iloc[0]
        oldest_close = daily_close_table.sort_index(ascending=True)['Daily Close'].iloc[0]
        newest_dividend = dividends_table.sort_index(ascending=False)['Dividends'].iloc[0]
        oldest_dividend = dividends_table.sort_index(ascending=True)['Dividends'].iloc[0]
        
        # Calculate growth rates
        HPR = (newest_dividend - oldest_dividend) / oldest_dividend
        newest_date = dividends_table.sort_index(ascending=False).index[0]
        oldest_date = dividends_table.sort_index(ascending=True).index[0]
        years_diff = (newest_date - oldest_date).days / 365.25
        Annum = (1 + HPR) ** (1 / years_diff) - 1
        
        # Create text widget
        text_widget = tk.Text(self.dividend_text_frame, wrap=tk.WORD, padx=10, pady=10)
        text_widget.pack(fill=tk.BOTH, expand=True)
        
        # Format and insert the information
        info_text = f"""
Dividend Analysis for {stock}:
Current Annual Dividend: ${newest_dividend*4:.2f}
Dividend Yield (Current Price): {(newest_dividend*4 / newest_close * 100):.2f}%
Dividend Yield (Purchace at Innital): {(newest_dividend*4 / oldest_close * 100):.2f}%
Total Dividend Growth: {HPR * 100:.2f}%
Annualized Dividend Growth: {Annum * 100:.2f}%

"""
        text_widget.insert(tk.END, info_text)
        text_widget.config(state=tk.DISABLED)  # Make text read-only

if __name__ == "__main__":
    root = tk.Tk()
    app = StockAnalysisApp(root)
    root.mainloop()