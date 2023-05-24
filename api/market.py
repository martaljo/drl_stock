import requests

access_key = #Add API key here
eod_ep = "http://api.marketstack.com/v1/eod"

def retrieve_historical(symbols, date_from, date_to):
	params = {
		"access_key": access_key,
		"symbols": symbols,
		"date_to": date_to,
		"date_from": date_from
	}
	api_result = requests.get(eod_ep, params)

	return api_result.json()['data']

# print(retrieve_historical("AAPL,UAL,BABA", "2023-05-14", "2023-05-18"))