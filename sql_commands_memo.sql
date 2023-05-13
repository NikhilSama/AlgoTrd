
select * from niftystratview  order by sharpe asc limit 10;
select * from niftystratview  order by cast(retrn as decimal) desc limit 10;
select count(*) from PerfNiftyAllStrat;

ALTER TABLE PerfNiftyRenko ADD COLUMN id INT primary key auto_increment;

CREATE VIEW renkoview AS 
SELECT id as id,
        num_trades,
        cfgTicker as t,
       FORMAT(max_drawdown_from_prev_peak_sum*100,2) AS drawdn,
       FORMAT(`return`*100,2) AS retrn,
       ROUND(sharpe_ratio,2) AS sharpe,
        FORMAT(average_per_trade_return*100,2) AS avgRet,
        FORMAT(std_dev_pertrade_return*100,2) AS stdDev,
        ROUND(skewness_pertrade_return,1) AS skew,
        ROUND(kurtosis_pertrade_return,1) AS kurtosis,
        FORMAT(avg_daily_return*100,2) AS dayAv,
        ROUND(sharpe_daily_return,2) AS dayShrp,
        FORMAT(std_daily_return*100,2) AS dayStd,
        ROUND(skew_daily_return,1) AS daySkew,
        ROUND(kurtosis_daily_return,1) AS dayKurt,
        maLen as maLen,
       bandWidth as bw,
       cfgMiniBandWidthMult as miBW,
       cfgSuperBandWidthMult as maBW,
       fastMALen as fstMA,
       adxLen,
       adxThresh,
       adxThreshYellowMultiplier as axdMult,
       numCandlesForSlopeProjection as candles,
       atrLen,
       ma_slope_thresh as slpThres,
       ma_slope_thresh_yellow_multiplier as slpMult,
        signalGenerators as sg,
        duration_in_days as days,
        startTime,
        endTime 
FROM PerfNiftyRenko;

CREATE VIEW renkoview AS 
SELECT id as id,
        num_trades,
       FORMAT(max_drawdown_from_prev_peak_sum*100,2) AS drawdn,
       FORMAT(`return`*100,2) AS retrn,
       ROUND(sharpe_ratio,2) AS sharpe,
       ROUND(calamar_ratio,2) AS calamar,
        FORMAT(average_per_trade_return*100,2) AS avgRet,
        FORMAT(std_dev_pertrade_return*100,2) AS stdDev,
        ROUND(skewness_pertrade_return,1) AS skew,
        ROUND(kurtosis_pertrade_return,1) AS kurtosis,
        FORMAT(avg_daily_return*100,2) AS dayAv,
        ROUND(sharpe_daily_return,2) AS dayShrp,
        FORMAT(std_daily_return*100,2) AS dayStd,
        ROUND(skew_daily_return,1) AS daySkew,
        ROUND(kurtosis_daily_return,1) AS dayKurt,
        atrLen,
        cfgRenkoBrickMultiplier as brick_mult,
        signalGenerators as sg,
        duration_in_days as days,
        startTime,
        endTime 
FROM PerfNiftyRenko;

create view cfgv5 as select id,maLen,bw,fstMA,atrLen, adxLen,adxThresh,axdMult,candles,slpThres,slpMult \
from pviewv5;


CREATE INDEX perf_nifty_all_strat_idx ON PerfNiftyAllStrat(
  startTime,
  endTime,
  sharpe_ratio,
  max_drawdown_from_prev_peak_sum,
  `return`
);


select id,t,startTime,endTime,num_trades,drawdn,retrn,sharpe,avgRet, \
stdDev,dayAv,dayShrp,maLen,slpThres,adxLen,adxThresh,candles \
from niftystratview \
where startTime = '2022-05-09' \
order by sharpe desc, cast(retrn as decimal) desc ;


SELECT p.*
FROM (
    SELECT *,
           RANK() OVER (PARTITION BY startTime ORDER BY sharpe DESC, retrn desc) AS sharpe_rank
    FROM (
        SELECT         id as id,
        num_trades,
       FORMAT(max_drawdown_from_prev_peak_sum*100,2) AS drawdn,
       FORMAT(`return`*100,2) AS retrn,
       ROUND(sharpe_ratio,2) AS sharpe,
        FORMAT(average_per_trade_return*100,2) AS avgRet,
        FORMAT(std_dev_pertrade_return*100,2) AS stdDev,
        FORMAT(avg_daily_return*100,2) AS dayAv,
        ROUND(sharpe_daily_return,2) AS dayShrp,
        FORMAT(std_daily_return*100,2) AS dayStd,
        maLen as maLen,
       adxLen,
       adxThresh,
       numCandlesForSlopeProjection as candles,
       ma_slope_thresh as slpThres,
        signalGenerators as sg,
        duration_in_days as days,
        startTime,
        endTime 

        FROM PerfNiftyAllStrat
        WHERE duration_in_days BETWEEN 27 AND 32 and adxLen=15 and ma_slope_thresh=0.1 and numCandlesForSlopeProjection=5 and maLen=6 and atrLen = 7 
    ) AS filtered
) AS p
WHERE p.sharpe_rank = 1;

