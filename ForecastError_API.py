#!/usr/bin/env python3

import numpy as np
from numpy import random
from flask import Flask
import pandas as pd
import plotly.express as px


def _safe_div(a, b):
    with np.errstate(divide='ignore', invalid='ignore'):
        c = np.true_divide(a, b)
        c[c == np.inf] = 0
        c = np.nan_to_num(c)
    return c


def _error(actual: np.ndarray, predicted: np.ndarray):
    """ Simple error """
    #    return actual - predicted
    return predicted - actual


def _percentage_error(actual: np.ndarray, predicted: np.ndarray):
    """
    Percentage error
    Note: result is NOT multiplied by 100
    """
    return _safe_div(_error(actual, predicted), actual)


def _naive_forecasting(actual: np.ndarray, seasonality: int = 1):
    """ Naive forecasting method which just repeats previous samples """
    return actual[:-seasonality]


def _relative_error(actual: np.ndarray, predicted: np.ndarray, benchmark: np.ndarray = None):
    """ Relative Error """
    if benchmark is None or isinstance(benchmark, int):
        # If no benchmark prediction provided - use naive forecasting
        if not isinstance(benchmark, int):
            seasonality = 1
        else:
            seasonality = benchmark

        #        return _error(actual[seasonality:], predicted[seasonality:]) /\
        #               (_error(actual[seasonality:], _naive_forecasting(actual, seasonality)) + EPSILON)

        return _safe_div(_error(actual[seasonality:], predicted[seasonality:]),
                         _error(actual[seasonality:], _naive_forecasting(actual, seasonality)))

    #    return _error(actual, predicted) / (_error(actual, benchmark) + EPSILON)
    return _safe_div(_error(actual, predicted), _error(actual, benchmark))


def _bounded_relative_error(actual: np.ndarray, predicted: np.ndarray, benchmark: np.ndarray = None):
    """ Bounded Relative Error """
    if benchmark is None or isinstance(benchmark, int):
        # If no benchmark prediction provided - use naive forecasting
        if not isinstance(benchmark, int):
            seasonality = 1
        else:
            seasonality = benchmark

        abs_err = np.abs(_error(actual[seasonality:], predicted[seasonality:]))
        abs_err_bench = np.abs(_error(actual[seasonality:], _naive_forecasting(actual, seasonality)))
    else:
        abs_err = np.abs(_error(actual, predicted))
        abs_err_bench = np.abs(_error(actual, benchmark))

    #    return abs_err / (abs_err + abs_err_bench + EPSILON)
    return _safe_div(abs_err, (abs_err + abs_err_bench))


def _geometric_mean(a, axis=0, dtype=None):
    """ Geometric mean """
    if not isinstance(a, np.ndarray):  # if not an ndarray object attempt to convert it
        log_a = np.log(np.array(a, dtype=dtype))
    elif dtype:  # Must change the default dtype allowing array type
        if isinstance(a, np.ma.MaskedArray):
            log_a = np.log(np.ma.asarray(a, dtype=dtype))
        else:
            log_a = np.log(np.asarray(a, dtype=dtype))
    else:
        log_a = np.log(a)
    return np.exp(log_a.mean(axis=axis))


def mse(actual: np.ndarray, predicted: np.ndarray):
    """ Mean Squared Error """
    return np.mean(np.square(_error(actual, predicted)))


def rmse(actual: np.ndarray, predicted: np.ndarray):
    """ Root Mean Squared Error """
    return np.sqrt(mse(actual, predicted))


def nrmse(actual: np.ndarray, predicted: np.ndarray):
    """ Normalized Root Mean Squared Error """
    # return rmse(actual, predicted) / (actual.max() - actual.min())
    return rmse(actual, predicted) / (np.max(actual) - np.min(actual))


def me(actual: np.ndarray, predicted: np.ndarray):
    """ Mean Error """
    return np.mean(_error(actual, predicted))


def mae(actual: np.ndarray, predicted: np.ndarray):
    """ Mean Absolute Error """
    return np.mean(np.abs(_error(actual, predicted)))


mad = mae  # Mean Absolute Deviation (it is the same as MAE)


def gmae(actual: np.ndarray, predicted: np.ndarray):
    """ Geometric Mean Absolute Error """
    return _geometric_mean(np.abs(_error(actual, predicted)))


def mdae(actual: np.ndarray, predicted: np.ndarray):
    """ Median Absolute Error """
    return np.median(np.abs(_error(actual, predicted)))


def mpe(actual: np.ndarray, predicted: np.ndarray):
    """ Mean Percentage Error """
    return np.mean(_percentage_error(actual, predicted))


def mape(actual: np.ndarray, predicted: np.ndarray):
    """
    Mean Absolute Percentage Error
    Properties:
        + Easy to interpret
        + Scale independent
        - Biased, not symmetric
        - Undefined when actual[t] == 0
    Note: result is NOT multiplied by 100
    """
    return np.mean(np.abs(_percentage_error(actual, predicted)))


def mdape(actual: np.ndarray, predicted: np.ndarray):
    """
    Median Absolute Percentage Error
    Note: result is NOT multiplied by 100
    """
    return np.median(np.abs(_percentage_error(actual, predicted)))


def smape(actual: np.ndarray, predicted: np.ndarray):
    """
    Symmetric Mean Absolute Percentage Error
    Note: result is NOT multiplied by 100
    """
    #    return np.mean(2.0 * np.abs(actual - predicted) / ((np.abs(actual) + np.abs(predicted)) + EPSILON))
    return np.mean(2.0 * (_safe_div(np.abs(actual - predicted), (np.abs(actual) + np.abs(predicted)))))


def smdape(actual: np.ndarray, predicted: np.ndarray):
    """
    Symmetric Median Absolute Percentage Error
    Note: result is NOT multiplied by 100
    """
    #    return np.median(2.0 * np.abs(actual - predicted) / ((np.abs(actual) + np.abs(predicted)) + EPSILON))
    return np.median(2.0 * (_safe_div(np.abs(actual - predicted), (np.abs(actual) + np.abs(predicted)))))


def bias_tracking(actual: np.ndarray, predicted: np.ndarray):
    return np.mean(_safe_div((actual - predicted), np.abs(actual - predicted)))


def bias_nfm(actual: np.ndarray, predicted: np.ndarray):
    return np.mean(_safe_div((actual - predicted), (actual + predicted)))


def maape(actual: np.ndarray, predicted: np.ndarray):
    """
    Mean Arctangent Absolute Percentage Error
    Note: result is NOT multiplied by 100
    """
    #    return np.mean(np.arctan(np.abs((actual - predicted) / (actual + EPSILON))))
    return np.mean(np.arctan(np.abs(_safe_div((actual - predicted), actual))))


def mase(actual: np.ndarray, predicted: np.ndarray, seasonality: int = 1):
    """
    Mean Absolute Scaled Error
    Baseline (benchmark) is computed with naive forecasting (shifted by @seasonality)
    """
    return mae(actual, predicted) / mae(actual[seasonality:], _naive_forecasting(actual, seasonality))


def std_ae(actual: np.ndarray, predicted: np.ndarray):
    """ Normalized Absolute Error """
    __mae = mae(actual, predicted)
    return np.sqrt(np.sum(np.square(_error(actual, predicted) - __mae)) / (len(actual) - 1))


def std_ape(actual: np.ndarray, predicted: np.ndarray):
    """ Normalized Absolute Percentage Error """
    __mape = mape(actual, predicted)
    return np.sqrt(np.sum(np.square(_percentage_error(actual, predicted) - __mape)) / (len(actual) - 1))


def rmspe(actual: np.ndarray, predicted: np.ndarray):
    """
    Root Mean Squared Percentage Error
    Note: result is NOT multiplied by 100
    """
    return np.sqrt(np.mean(np.square(_percentage_error(actual, predicted))))


def rmdspe(actual: np.ndarray, predicted: np.ndarray):
    """
    Root Median Squared Percentage Error
    Note: result is NOT multiplied by 100
    """
    return np.sqrt(np.median(np.square(_percentage_error(actual, predicted))))


def rmsse(actual: np.ndarray, predicted: np.ndarray, seasonality: int = 1):
    """ Root Mean Squared Scaled Error """
    q = np.abs(_error(actual, predicted)) / mae(actual[seasonality:], _naive_forecasting(actual, seasonality))
    return np.sqrt(np.mean(np.square(q)))


def inrse(actual: np.ndarray, predicted: np.ndarray):
    """ Integral Normalized Root Squared Error """
    return np.sqrt(np.sum(np.square(_error(actual, predicted))) / np.sum(np.square(actual - np.mean(actual))))


def rrse(actual: np.ndarray, predicted: np.ndarray):
    """ Root Relative Squared Error """
    return np.sqrt(np.sum(np.square(actual - predicted)) / np.sum(np.square(actual - np.mean(actual))))


def mre(actual: np.ndarray, predicted: np.ndarray, benchmark: np.ndarray = None):
    """ Mean Relative Error """
    return np.mean(_relative_error(actual, predicted, benchmark))


def rae(actual: np.ndarray, predicted: np.ndarray):
    """ Relative Absolute Error (aka Approximation Error) """
    #    return np.sum(np.abs(actual - predicted)) / (np.sum(np.abs(actual - np.mean(actual))) + EPSILON)
    return _safe_div(np.sum(np.abs(actual - predicted)), np.sum(np.abs(actual - np.mean(actual))))


def mrae(actual: np.ndarray, predicted: np.ndarray, benchmark: np.ndarray = None):
    """ Mean Relative Absolute Error """
    return np.mean(np.abs(_relative_error(actual, predicted, benchmark)))


def mdrae(actual: np.ndarray, predicted: np.ndarray, benchmark: np.ndarray = None):
    """ Median Relative Absolute Error """
    return np.median(np.abs(_relative_error(actual, predicted, benchmark)))


def gmrae(actual: np.ndarray, predicted: np.ndarray, benchmark: np.ndarray = None):
    """ Geometric Mean Relative Absolute Error """
    return _geometric_mean(np.abs(_relative_error(actual, predicted, benchmark)))


def mbrae(actual: np.ndarray, predicted: np.ndarray, benchmark: np.ndarray = None):
    """ Mean Bounded Relative Absolute Error """
    return np.mean(_bounded_relative_error(actual, predicted, benchmark))


def umbrae(actual: np.ndarray, predicted: np.ndarray, benchmark: np.ndarray = None):
    """ Unscaled Mean Bounded Relative Absolute Error """
    __mbrae = mbrae(actual, predicted, benchmark)
    return __mbrae / (1 - __mbrae)


def mda(actual: np.ndarray, predicted: np.ndarray):
    """ Mean Directional Accuracy """
    # return np.mean((np.sign(actual[1:] - actual[:-1]) == np.sign(predicted[1:] - predicted[:-1])).astype(int))
    return np.mean((np.sign(actual[1:] - actual[:-1]) == np.sign(predicted[1:] - predicted[:-1])))


def wmape(actual: np.ndarray, predicted: np.ndarray):
    """
    Weighted Mean Absolute Percentage Error
    Note: result is NOT multiplied by 100
    """
    se_actual_prod_mape = actual * _safe_div(abs(actual - predicted), actual)
    return se_actual_prod_mape.sum() / actual.sum()


app = Flask(__name__)


@app.route('/')
def chart():
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
        'error': _error,
        '%_error': _percentage_error,
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
        'error': 'Simple Error',
        '%_error': 'Percentage Error',
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

    num_iterations = 5

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

    for x in range(1, num_iterations + 1):
        print('START: {}'.format(x))
        tests = 1
        # df = {}
        df = {"Test": []}

        for test_values in test_values_set:
            print('Test: {} / {}'.format(tests, num_tests))

            print(f'{"FORECAST":<13} - low: {test_values[0][0]:>4} high: {test_values[0][1]:>4}'
                  f' range: {test_values[0][2]:>4}')
            print(f'{"VAR to ACTUAL":<13} - low: {test_values[1][0]:>4} high: {test_values[1][1]:>4}'
                  f' range: {test_values[1][2]:>4}')

            the_forecast = np.array(
                random.randint(low=test_values[0][0], high=test_values[0][1], size=test_values[0][2]))

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

        df.update(_df)
        df_list.append(df)

        for key, value in df.items():
            print(key, ' : ', value)

        print('')
        print('FINISH: {}'.format(x))
        print('')

        plot_data = pd.DataFrame.from_dict(df)
        plot = px.line(plot_data, x='Test', y=['MAPE', 'MASE', 'MDAPE', 'WMAPE', 'SMAPE', 'NRMSE', 'MPE'],
                       title="Forecast Measurement Metrics - Run {}".format(x))
        plot.show()

    w = 1
    for thing in df_list:
        print(w)
        for key, value in thing.items():
            print(key, ' : ', value)
        print('')
        w += 1

    return ("drawing chart")


if __name__ == "__main__":
    app.run()
