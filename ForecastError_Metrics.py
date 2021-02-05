import numpy as np


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


# rmsd sum(square(_error(actual, predicted)))/number of observations
def rmsd(actual: np.ndarray, predicted: np.ndarray):
    """ Root Mean Squared Deviation"""
    one = np.square(_error(actual, predicted))
    return one.sum() / actual.size()
    # return np.sum(np.square(_error(actual, predicted))) / actual.size()
    # is this just RMSE?


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
    return 100 * np.mean(_percentage_error(actual, predicted))


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
    return 100 * np.mean(np.abs(_percentage_error(actual, predicted)))


def mdape(actual: np.ndarray, predicted: np.ndarray):
    """
    Median Absolute Percentage Error
    Note: result is NOT multiplied by 100
    """
    return 100 * np.median(np.abs(_percentage_error(actual, predicted)))


# there are 3 versions of this formula.  look at wiki page
def smape(actual: np.ndarray, predicted: np.ndarray):
    """
    Symmetric Mean Absolute Percentage Error
    Note: result is NOT multiplied by 100
    """
    #    return np.mean(2.0 * np.abs(actual - predicted) / ((np.abs(actual) + np.abs(predicted)) + EPSILON))
    return 100 * np.mean(2.0 * (_safe_div(np.abs(actual - predicted), (np.abs(actual) + np.abs(predicted)))))


def smdape(actual: np.ndarray, predicted: np.ndarray):
    """
    Symmetric Median Absolute Percentage Error
    Note: result is NOT multiplied by 100
    """
    #    return np.median(2.0 * np.abs(actual - predicted) / ((np.abs(actual) + np.abs(predicted)) + EPSILON))
    return 100 * np.median(2.0 * (_safe_div(np.abs(actual - predicted), (np.abs(actual) + np.abs(predicted)))))


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
    return 100 * np.mean(np.arctan(np.abs(_safe_div((actual - predicted), actual))))


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
    return 100 * np.sqrt(np.mean(np.square(_percentage_error(actual, predicted))))


def rmdspe(actual: np.ndarray, predicted: np.ndarray):
    """
    Root Median Squared Percentage Error
    Note: result is NOT multiplied by 100
    """
    return 100 * np.sqrt(np.median(np.square(_percentage_error(actual, predicted))))


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


def wape(actual: np.ndarray, predicted: np.ndarray):
    """
    Weighted Average Percentage Error
    Note: result is NOT multiplied by 100
    """
    se_actual_prod_mape = actual * _safe_div(abs(actual - predicted), actual)
    return 100 * se_actual_prod_mape.sum() / actual.sum()


wmape = wape  # Weighted Mead Absolute Percentage Error = WAPE when the weighting factor is the actuals

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
    'WAPE': wape,
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
    'MBREA': mbrae,
    'UMBRAE': umbrae,
    'MDA': mda,
    'WMAPE': wmape,
    'RMSD': rmsd,
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
    'WAPE': 'Weighted Average Percentage Error',
    'MDAPE': 'Median Absolute Percentage Error',
    'SMAPE': 'Symmetric Mean Absolute Percentage Error',
    'SMDAPE': 'Symmetric Median Absolute Percentage Error',
    'MAAPE': 'Mean Arctangent Absolute Percentage Error',
    'MASE': 'Mean Absolute Scaled Error',
    'STD_AE': 'Normalized Absolute Error',
    'STD_APE': 'Normalized Absolute Percentage Error',
    'RMSPE': 'Root Mean Squared Percentage Error',
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
    'RMSD': 'Root Mean Squared Deviation',
    'BIAS_TRACK': 'Bias Tracking Signal',
    'BIAS_NFM': 'Bias Normalized Forecast Metric',
}
