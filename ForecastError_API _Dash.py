#!/usr/bin/env python3

import flask
import dash
import dash_table
import dash_core_components as dcc
import dash_html_components as html

from numpy import random
from flask import Flask
import pandas as pd
import plotly.express as px
from ForecastError_Metrics import *
import datetime
import os

server = flask.Flask('app')
server.secret_key = os.environ.get('secret_key', 'secret')

app = dash.Dash('app', server=server)


def current_date():
    now = datetime.datetime.now()
    the_text = '{}'.format(now.strftime("%Y-%m-%d %H:%M:%S"))
    return html.P(children=the_text)


def forecasterror(iters=1):
    test_values_set = [
        [[10, 1050, 500], [-50, 100, 500]],
        [[5, 505, 500], [-5, 5, 500]],
        [[5, 1050, 1000], [-55, 100, 1000]],
        [[100, 3100, 1000], [-200, 500, 1000]],
        [[100, 3100, 1000], [-500, 500, 1000]],
        [[100, 3100, 20000], [-200, 500, 20000]],
        [[100, 3100, 100], [-200, 500, 100]],
        [[5, 3100, 100], [-100, 500, 100]],
        [[5, 3100, 20000], [-100, 500, 20000]],
        [[5, 3100, 20000], [-500, 500, 20000]],
        [[5, 3100, 500], [-100, 500, 500]],
        [[5, 3100, 500], [-500, 500, 500]],
        [[5, 1000, 500], [-100, 300, 500]],
        [[5, 1000, 500], [-500, 300, 500]],
        [[5, 1000, 500], [10, 300, 500]],
        [[5, 1000, 500], [-300, -10, 500]],
        [[5, 1000, 500], [-200, -10, 500]],
        [[5, 6, 500], [-2, 2, 500]],
        [[5, 6, 500], [-7, 7, 500]],
        [[5, 6, 500], [-20, 20, 500]],
        [[5, 6, 500], [-50, 50, 500]]
    ]

    test_metrics = ['MPE', 'MAPE', 'WAPE', 'MDAPE', 'SMAPE', 'ME', 'MAE', 'MASE', 'MSE', 'RMSE', 'NRMSE',
                    'BIAS_TRACK', 'BIAS_NFM', 'NRMSE']

    def evaluate(actual: np.ndarray, predicted: np.ndarray, metrics=('MAPE', 'MASE')):
        results = {}
        for name in metrics:
            try:
                results[name] = METRICS[name](actual, predicted)
            except Exception as err:
                results[name] = np.nan
                print('Unable to compute metric {0}: {1}'.format(name, err))
        return results

    df_list = []
    output_array = []
    metrics_array = []

    num_tests = len(test_values_set)
    num_iterations = int(iters)
    num_iterations = 1

    for x in range(1, num_iterations + 1):
        the_text = 'START {}'.format(x)
        output_array.append(html.P(children=the_text))

        tests = 1
        df = {"Test": []}

        for test_values in test_values_set:
            the_text = 'Test: {} / {}'.format(tests, num_tests)
            output_array.append(html.Div(children=the_text))

            params_array = [["Forcast Range", test_values[0][0], test_values[0][1], test_values[0][2]],
                          ["VAR to Forecast Range", test_values[1][0], test_values[1][1], test_values[1][2]]]
            _df = pd.DataFrame(params_array, columns=['Data Set', 'Low', 'High', 'Size'])

            table_id = 'table-params-{}'.format(tests)
            output_array.append(html.Div(dash_table.DataTable(
                id=table_id,
                columns=[{"name": i, "id": i} for i in _df.columns],
                data=_df.to_dict('records'),

                style_data_conditional=[
                    {
                        'if': {'row_index': 'odd'},
                        'backgroundColor': 'rgb(248, 248, 248)'
                    }
                ],
                style_header={
                    'backgroundColor': 'rgb(230, 230, 230)',
                    'fontWeight': 'bold'
                }
            ), style={'width': '20%', 'display': 'inline-block'}))

            the_forecast = np.array(
                random.randint(low=test_values[0][0], high=test_values[0][1], size=test_values[0][2]))

            adjust = np.array(random.randint(low=test_values[1][0], high=test_values[1][1], size=test_values[1][2]))

            the_actuals = the_forecast + adjust
            the_actuals[the_actuals < 0] = 0

            # the_forecast = np.array([200,300,400,500])
            # the_actuals = np.array([230,290,740,450])

            # the_actuals = np.array([27.58,25.95,26.08,26.36,27.99,29.61,28.85,29.43,29.67,30.19,31.79,31.98])
            # the_forecast = np.array([27.58,26.765,26.015,26.220,27.175,28.8,29.23,29.14,29.55,29.93,30.99,31.885])

            the_text = 'Zeros: {}'.format(test_values[0][2] - np.count_nonzero(the_actuals))
            output_array.append(html.Div(children=the_text))

            answers = evaluate(the_actuals, the_forecast, metrics=test_metrics)

            for answer in answers:
                metrics_array.append([answer, '{:.2f}'.format(round(answers[answer], 2)), METRICS_NAME[answer]])

            # df["Test"].append(tests)

            _df = pd.DataFrame(metrics_array, columns=['Metric', 'Value', 'Description'])
            metrics_array = []

            table_id = 'table-metrics-{}-{}'.format(x, tests)
            # print(table_id)

            output_array.append(html.Div(dash_table.DataTable(
                id=table_id,
                columns=[{"name": i, "id": i} for i in _df.columns],
                data=_df.to_dict('records'),

                style_data_conditional=[
                    {
                        'if': {'row_index': 'odd'},
                        'backgroundColor': 'rgb(248, 248, 248)'
                    },
                    {
                        'if': {'column_id': 'Description'},
                        'textAlign': 'left'
                    }
                ],
                style_header_conditional=[
                    {
                        'if': {'column_id': 'Description'},
                        'textAlign': 'left'
                    }
                ],
                style_header={
                    'backgroundColor': 'rgb(230, 230, 230)',
                    'fontWeight': 'bold'
                }
            ), style={'width': '20%', 'display': 'inline-block'}))

            tests += 1

        the_text = 'FINISH {}'.format(x)
        output_array.append(html.Div(children=the_text))

        """
        df_list.append(df)

        # for key, value in df.items():
        #     print(key, ' : ', value)

        _df = pd.DataFrame.from_dict(df)
        output += _df.to_html(index=False, justify='left')


        plot_data = pd.DataFrame.from_dict(df)
        plot = px.line(plot_data, x='Test', y=['MAPE', 'MDAPE', 'WAPE', 'SMAPE', 'MPE'],
                       title="Forecast Measurement Metrics - Percent Error - Run {}".format(x))
        plot.show()

        plot = px.line(plot_data, x='Test', y=['MAE', 'RMSE'],
                       title="Forecast Measurement Metrics - Scaled Error - Run {}".format(x))
        plot.show()

        plot = px.line(plot_data, x='Test', y=['NRMSE', 'MASE'],
                       title="Forecast Measurement Metrics - Absolute Error - Run {}".format(x))
        plot.show()
        """

    """
    w = 1
    for thing in df_list:
        # print(w)
        output += '{}<br>'.format(w)

        # for key, value in thing.items():
        #    print(key, ' : ', value)

        _df = pd.DataFrame.from_dict(thing)
        output += _df.to_html(index=False, justify='left')

        # print('')
        output += '<br>'

        w += 1

    # not using DASH so had to do a little HTML tweaking for HTML characters
    output = output.replace('@@@', '>')
    output = output.replace('@@', '<')

    """

    # david_list = []
    # david_list.append(output_start)
    # david_list.extend(output_params)
    # david_list.extend(output_zeros)
    # david_list.extend(output_metrics)

    # return html.Div(children=[output_start, output_test_number])
    # return html.Div(children=david_list)
    return html.Div(children=output_array)

output_date = current_date()

app.layout = html.Div(children=[output_date, forecasterror(1)])

if __name__ == "__main__":
    server.run(host='localhost', port=5001, debug=True)
