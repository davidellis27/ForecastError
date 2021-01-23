#!/usr/bin/env python3

import numpy as np
import pandas as pd
from numpy import random


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
    return rmse(actual, predicted) / (actual.max() - actual.min())


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
    se_mape = _safe_div(abs(actual - predicted), actual)
    se_actual_prod_mape = actual * se_mape
    ft_actual_sum = actual.sum()
    ft_actual_prod_mape_sum = se_actual_prod_mape.sum()
    ft_wmape_forecast = ft_actual_prod_mape_sum / ft_actual_sum

    return ft_wmape_forecast


#    return np.mean(2.0 * (_safe_div(np.abs(actual - predicted), (np.abs(actual) + np.abs(predicted)))))


METRICS = {
    'mse': mse,
    'rmse': rmse,
    'nrmse': nrmse,
    'me': me,
    'mae': mae,
    'mad': mad,
    'gmae': gmae,
    'mdae': mdae,
    'mpe': mpe,
    'mape': mape,
    'mdape': mdape,
    'smape': smape,
    'smdape': smdape,
    'maape': maape,
    'mase': mase,
    'std_ae': std_ae,
    'std_ape': std_ape,
    'rmspe': rmspe,
    'rmdspe': rmdspe,
    'rmsse': rmsse,
    'inrse': inrse,
    'rrse': rrse,
    'mre': mre,
    'rae': rae,
    'mrae': mrae,
    'mdrae': mdrae,
    'gmrae': gmrae,
    'mbrae': mbrae,
    'umbrae': umbrae,
    'mda': mda,
    'wmape': wmape,
    'error': _error,
    '%_error': _percentage_error,
}

METRICS_NAME = {
    'mse': 'Mean Squared Error',
    'rmse': 'Root Mean Squared Error',
    'nrmse': 'Normalized Root Mean Squared Error',
    'me': 'Mean Error',
    'mae': 'Mean Absolute Error',
    'mad': 'Mean Absolute Deviation',
    'gmae': 'Geometric Mean Absolute Error',
    'mdae': 'Median Absolute Error',
    'mpe': 'Mean Percentage Error',
    'mape': 'Mean Absolute Percentage Error',
    'mdape': 'Median Absolute percentage Error',
    'smape': 'Symmetric Mean Absolute Percentage Error',
    'smdape': 'Symmetric Median Absolute Percentage Error',
    'maape': 'Mean Arctangent Absolute Percentage Error',
    'mase': 'Mean Absolute Scaled Error',
    'std_ae': 'Normalized Absolute Error',
    'std_ape': 'Normalized Absolute Percentage Error',
    'rmspe': 'Root Mean Squared Percentage Error',
    'rmdspe': 'Root Median Squared Percentage Error',
    'rmsse': 'Root Mean Squared Scaled Error',
    'inrse': 'Integral Normalized Root Squared Error',
    'rrse': 'Root Relative Squared Error',
    'mre': 'Mean Relative Error',
    'rae': 'Relative Absolute Error',
    'mrae': 'Mean Relative Absolute Error',
    'mdrae': 'Median Relative Absolute Error',
    'gmrae': 'Geometric Mean Relative Absolute Error',
    'mbrae': 'Mean Bounded Relative Absolute Error',
    'umbrae': 'Unscaled Mean Bounded Relative Absolute Error',
    'mda': 'Mean Directional Accuracy',
    'wmape': 'Weighted Mean Absolute Percentage Error',
    'error': 'Simple Error',
    '%_error': 'Percentage Error',
}


def evaluate(actual: np.ndarray, predicted: np.ndarray, metrics=('mape', 'mase')):
    results = {}
    for name in metrics:
        try:
            results[name] = METRICS[name](actual, predicted)
        except Exception as err:
            results[name] = np.nan
            print('Unable to compute metric {0}: {1}'.format(name, err))
    return results


test_values_set = [[[10, 1050, 500], [-50, 100, 500]],
                   [[5, 505, 500], [-5, 5, 500]]]

for test_values in test_values_set:
    print('{:<8} - low: {:>4} high: {:>4} range: {:>4}'.format('FORECAST', test_values[0][0], test_values[0][1],
                                                               test_values[0][2]))
    print('{:<8} - low: {:>4} high: {:>4} range: {:>4}'.format('ADJUST', test_values[1][0], test_values[1][1],
                                                               test_values[1][2]))
    print('')

    the_forecast = np.array(random.randint(low=10, high=1050, size=500))

    adjust = np.array(random.randint(low=-50, high=100, size=500))

    the_actuals = the_forecast + adjust
    the_actuals[the_actuals < 0] = 0

    answers = evaluate(the_actuals, the_forecast,
                       metrics=('mape', 'mase', 'mdape', 'wmape', 'smape', 'me', 'mae', 'mpe'))

    # print(the_forecast)
    # print(the_actuals)
    # print(the_actuals - the_forecast)

    for answer in answers:
        print(f'{answer:<8}{answers[answer] * 100:<20}{METRICS_NAME[answer]:<50}')

    print('\n')
