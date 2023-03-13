

use trading;
#select * from instruments_zerodha where tradingsymbol = 'HDFCLIFE23MAR490CE'
#select * from ohlcv1m where symbol = 'ADANIENT' order by date desc
#select * from ohlcv1m where date > '2023-03-13 12:17:00' and symbol = 'ITC'
#select date from ohlcv1m order by date desc limit 1

#delete from ohlcv1m where date >= '2023-03-13 13:37:00'
#show processlist;
select * from ohlcv1m where date >= '2023-03-13 15:07:00'
#delete from ohlcv1m where date >= '2023-03-13 15:07:00'