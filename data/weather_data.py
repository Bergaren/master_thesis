from wwo_hist import retrieve_hist_data

frequency = 24
start_date = '1-Jan-2015'
end_date = '1-Jan-2021'
api_key = 'd8973603d0934f34a22105551210802'
location_list = ['malmo','goteborg','linkoping', 'stockholm']

hist_weather_data = retrieve_hist_data(api_key,
                                location_list,
                                start_date,
                                end_date,
                                frequency,
                                location_label = False,
                                export_csv = True,
                                store_df = True)