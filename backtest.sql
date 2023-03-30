CREATE TABLE backtest (
    id INT NOT NULL AUTO_INCREMENT,
    trading_days FLOAT NOT NULL,
    days_in_trade FLOAT NOT NULL,
    num_trades FLOAT NOT NULL,
    num_winning_trades FLOAT NOT NULL,
    num_losing_trades FLOAT NOT NULL,
    win_pct FLOAT NOT NULL,
    ret FLOAT NOT NULL,
    ret_per_day_in_trade FLOAT NOT NULL,
    annualized_ret FLOAT NOT NULL,
    avg_per_trade_return FLOAT NOT NULL,
    avg_of_per_ticker_std_dev_across_trades FLOAT NOT NULL,
    skewness_pertrade_return FLOAT,
    kurtosis_pertrade_return FLOAT,
    wins FLOAT NOT NULL,
    loss FLOAT NOT NULL,
    std_dev_across_stocks FLOAT NOT NULL,
    kurtosis_across_stocks FLOAT NOT NULL,
    skewness_across_stocks FLOAT NOT NULL,
    maLen INT NOT NULL,
    bandWidth FLOAT NOT NULL,
    source_filename VARCHAR(255) NOT NULL,
    created_time DATETIME NOT NULL,
    PRIMARY KEY (id),
    INDEX (trading_days),
    INDEX (days_in_trade),20 + 2 +1 index
    INDEX (num_trades),
    INDEX (num_winning_trades),
    INDEX (num_losing_trades),
    INDEX (win_pct),
    INDEX (ret),
    INDEX (ret_per_day_in_trade),
    INDEX (annualized_ret),
    INDEX (avg_per_trade_return),
    INDEX (avg_of_per_ticker_std_dev_across_trades),
    INDEX (skewness_pertrade_return),
    INDEX (kurtosis_pertrade_return),
    INDEX (wins),
    INDEX (loss),
    INDEX (std_dev_across_stocks),
    INDEX (kurtosis_across_stocks),
    INDEX (skewness_across_stocks),
    INDEX (maLen),
    INDEX (bandWidth),
    INDEX (source_filename),
    INDEX (created_time)
);
