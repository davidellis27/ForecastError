#!/usr/bin/env python3

from numpy import random
from ForecastError_Metrics import *
import datetime


METRICS = {
    'MSE': mse,
    'RMSE': rmse,
    'NRMSE': nrmse,
    'ME': me,
    'MAE': mae,
    'MAD': mad,
    'GMAE': gmae,
    'MDAE': mdae,
    'MPE': mpe,
    'MAPE': mape,
    'MDAPE': mdape,
    'SMAPE': smape,
    'SMDAPE': smdape,
    'MAAPE': maape,
    'MASE': mase,
    'STD_AE': std_ae,
    'STD_APE': std_ape,
    'RMSPE': rmspe,
    'RMDSPE': rmdspe,
    'RMSSE': rmsse,
    'INRSE': inrse,
    'RRSE': rrse,
    'MRE': mre,
    'RAE': rae,
    'MRAE': mrae,
    'MDRAE': mdrae,
    'GMRAE': gmrae,
    'mbrae': mbrae,
    'umbrae': umbrae,
    'MDA': mda,
    'WMAPE': wmape,
    'BIAS_TRACK': bias_tracking,
    'BIAS_NFM': bias_nfm,
}

METRICS_NAME = {
    'MSE': 'Mean Squared Error',
    'RMSE': 'Root Mean Squared Error',
    'NRMSE': 'Normalized Root Mean Squared Error',
    'ME': 'Mean Error',
    'MAE': 'Mean Absolute Error',
    'MAD': 'Mean Absolute Deviation',
    'GMAE': 'Geometric Mean Absolute Error',
    'MDAE': 'Median Absolute Error',
    'MPE': 'Mean Percentage Error',
    'MAPE': 'Mean Absolute Percentage Error',
    'MDAPE': 'Median Absolute Percentage Error',
    'SMAPE': 'Symmetric Mean Absolute Percentage Error',
    'SMDAPE': 'Symmetric Median Absolute Percentage Error',
    'MAAPE': 'Mean Arctangent Absolute Percentage Error',
    'MASE': 'Mean Absolute Scaled Error',
    'STD_AE': 'Normalized Absolute Error',
    'STD_APE': 'Normalized Absolute Percentage Error',
    'rmspe': 'Root Mean Squared Percentage Error',
    'RMDSPE': 'Root Median Squared Percentage Error',
    'RMSSE': 'Root Mean Squared Scaled Error',
    'IINRSE': 'Integral Normalized Root Squared Error',
    'RRSE': 'Root Relative Squared Error',
    'MRE': 'Mean Relative Error',
    'RAE': 'Relative Absolute Error',
    'MRAE': 'Mean Relative Absolute Error',
    'MDRAE': 'Median Relative Absolute Error',
    'GMRAE': 'Geometric Mean Relative Absolute Error',
    'MBRAE': 'Mean Bounded Relative Absolute Error',
    'UMBRAE': 'Unscaled Mean Bounded Relative Absolute Error',
    'MDA': 'Mean Directional Accuracy',
    'WMAPE': 'Weighted Mean Absolute Percentage Error',
    'BIAS_TRACK': 'Bias Tracking Signal',
    'BIAS_NFM': 'Bias Normalized Forecast Metric',
}


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

test_metrics = ['MAPE', 'MASE', 'MDAPE', 'WMAPE', 'SMAPE', 'ME', 'MAE', 'MPE', 'RMSE', 'NRMSE',
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

num_tests = len(test_values_set)
num_iterations = 4

now = datetime.datetime.now()
print(now.strftime("%Y-%m-%d %H:%M:%S"))
print('')

for x in range(1, num_iterations + 1):
    print('START: {}'.format(x))

    tests = 1
    df = {"Test": []}

    for test_values in test_values_set:
        print('Test: {} / {}'.format(tests, num_tests))

        print(f'{"FORECAST":<13} - low: {test_values[0][0]:>4} high: {test_values[0][1]:>4}'
              f' range: {test_values[0][2]:>4}')
        print(f'{"VAR to ACTUAL":<13} - low: {test_values[1][0]:>4} high: {test_values[1][1]:>4}'
              f' range: {test_values[1][2]:>4}')

        the_forecast = np.array(random.randint(low=test_values[0][0], high=test_values[0][1], size=test_values[0][2]))

        adjust = np.array(random.randint(low=test_values[1][0], high=test_values[1][1], size=test_values[1][2]))

        the_actuals = the_forecast + adjust
        the_actuals[the_actuals < 0] = 0

        print('Zeros: {}'.format(test_values[0][2] - np.count_nonzero(the_actuals)))
        print('')

        answers = evaluate(the_actuals, the_forecast, metrics=test_metrics)

        for answer in answers:
            print(f'{answer:<11}{answers[answer] * 100:>10.2f}  {METRICS_NAME[answer]:<}')

            if tests == 1:
                _df = {answer: [round(answers[answer] * 100, 2)]}
                df.update(_df)
                _df.clear()
            else:
                df[answer].append(round(answers[answer] * 100, 2))

        df["Test"].append(tests)

        tests += 1
        print('')

    df_list.append(df)

    for key, value in df.items():
        print(key, ' : ', value)

    print('')
    print('FINISH: {}'.format(x))
    print('')

w = 1
for thing in df_list:
    print(w)

    for key, value in thing.items():
        print(key, ' : ', value)
    print('')

    w += 1
