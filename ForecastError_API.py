#!/usr/bin/env python3

from numpy import random
from flask import Flask
import pandas as pd
import plotly.express as px
from ForecastError_Metrics import *
import datetime


app = Flask(__name__)


@app.route('/forecasterror/', methods=['GET'])
@app.route('/forecasterror/<iters>', methods=['GET'])
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

    test_metrics = ['MAPE', 'MASE', 'WAPE', 'MDAPE', 'SMAPE', 'ME', 'MAE', 'MPE', 'RMSE', 'NRMSE',
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
    output = ""
    output_array = []

    num_tests = len(test_values_set)
    num_iterations = int(iters)

    now = datetime.datetime.now()
    # print(now.strftime("%Y-%m-%d %H:%M:%S"))
    # print('')
    output += '{}<br><br>'.format(now.strftime("%Y-%m-%d %H:%M:%S"))

    for x in range(1, num_iterations + 1):
        # print('START: {}'.format(x))
        output += 'START {}<br>'.format(x)

        tests = 1
        df = {"Test": []}

        for test_values in test_values_set:
            # print('Test: {} / {}'.format(tests, num_tests))
            output += 'Test: {} / {}<br>'.format(tests, num_tests)

            # print(f'{"FORECAST":<13} - low: {test_values[0][0]:>4} high: {test_values[0][1]:>4}'
            #      f' range: {test_values[0][2]:>4}')
            output_array.append(["Forcast Range", test_values[0][0], test_values[0][1], test_values[0][2]])

            # print(f'{"VAR to ACTUAL":<13} - low: {test_values[1][0]:>4} high: {test_values[1][1]:>4}'
            #      f' range: {test_values[1][2]:>4}')
            output_array.append(["VAR to Forecast Range", test_values[1][0], test_values[1][1], test_values[1][2]])

            _df = pd.DataFrame(output_array, columns=['Data Set', 'Low', 'High', 'Size'])
            output_array = []

            output += _df.to_html(index=False, justify='left').replace('class="dataframe"',
                                                                       'style="border-collapse:collapse" '
                                                                       'class="dataframe"')

            the_forecast = np.array(
                random.randint(low=test_values[0][0], high=test_values[0][1], size=test_values[0][2]))

            adjust = np.array(random.randint(low=test_values[1][0], high=test_values[1][1], size=test_values[1][2]))

            the_actuals = the_forecast + adjust
            the_actuals[the_actuals < 0] = 0

            # print('Zeros: {}'.format(test_values[0][2] - np.count_nonzero(the_actuals)))
            output += 'Zeros: {}<br>'.format(test_values[0][2] - np.count_nonzero(the_actuals))
            # print('')
            output += '<br>'

            answers = evaluate(the_actuals, the_forecast, metrics=test_metrics)

            for answer in answers:
                # print(f'{answer:<11}{answers[answer] * 100:>10.2f}  {METRICS_NAME[answer]:<}')
                output_array.append(
                    [answer, '@@div style="text-align:right"@@@{:.2f}@@/div@@@'.format(round(answers[answer] * 100, 2)),
                     METRICS_NAME[answer]])

                if tests == 1:
                    _df = {answer: [round(answers[answer] * 100, 2)]}
                    df.update(_df)
                    _df.clear()
                else:
                    df[answer].append(round(answers[answer] * 100, 2))

            df["Test"].append(tests)

            tests += 1

            _df = pd.DataFrame(output_array, columns=['Metric', 'Value', 'Description'])
            output_array = []

            output += _df.to_html(index=False, justify='left').replace('class="dataframe"',
                                                                       'style="border-collapse:collapse" '
                                                                       'class="dataframe"')

            # print('')
            output += '<br>'

        df_list.append(df)

        # for key, value in df.items():
        #     print(key, ' : ', value)

        _df = pd.DataFrame.from_dict(df)
        output += _df.to_html(index=False, justify='left')

        # print('')
        # print('FINISH: {}'.format(x))
        # print('')
        output += '<br>'
        output += 'FINISH: {}<br>'.format(x)
        output += '<br>'

        plot_data = pd.DataFrame.from_dict(df)
        plot = px.line(plot_data, x='Test', y=['MAPE', 'MASE', 'MDAPE', 'WAPE', 'SMAPE', 'NRMSE', 'MPE'],
                       title="Forecast Measurement Metrics - Run {}".format(x))
        plot.show()

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

    return output


if __name__ == "__main__":
    app.run(host='localhost', port=5001)
