all: market q-table

market:
	perl build-market-input.pl market.csv data-in/gdp.csv data-in/manufacture-orders.csv data-in/manufacture-durable.csv break data-in/spy.csv

q-table:
	perl build-q-table.pl q-out.csv market.csv

less:
	less market.csv
