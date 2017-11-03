import AllChecks as qc
import numpy as np
import pandas as pd
import os
import time


def load_expected_result_table():
    """
    Use folders with sorted samples (Good, No_beads, Premature_Full, ...) to construct expected result and image paths
    folder list.

    :return: Tuple with pandas data frame with expected/correct result and list of image folders
    """

    # Define main paths, categories and their results
    main_path = '/media/anders/-Anders-5-/BluSense/ImageLibrary_Plasma'
    category_folders = ['Good', 'No_Beads', 'No_Reagents', 'No_Sample', 'Not_Filled', 'Premature_Partial',
                        'Premature_Full']
    category_results = {'Good': {'BcSpot': True, 'MescSpot': True, 'BloodPresent': np.NaN, 'MescProblem': False,
                                 'Error': '', 'Version': np.NaN},
                        'No_Beads': {'BcSpot': True, 'MescSpot': False, 'BloodPresent': np.NaN, 'MescProblem': False,
                                     'Error': '', 'Version': np.NaN},
                        'No_Reagents': {'BcSpot': False, 'MescSpot': True, 'BloodPresent': np.NaN, 'MescProblem': False,
                                        'Error': '', 'Version': np.NaN},
                        'No_Sample': {'BcSpot': False, 'MescSpot': False, 'BloodPresent': np.NaN, 'MescProblem': False,
                                      'Error': '', 'Version': np.NaN},
                        'Not_Filled': {'BcSpot': True, 'MescSpot': True, 'BloodPresent': np.NaN, 'MescProblem': True,
                                       'Error': '', 'Version': np.NaN},
                        'Premature_Full': {'BcSpot': True, 'MescSpot': np.NaN, 'BloodPresent': np.NaN,
                                           'MescProblem': False, 'Error': '', 'Version': np.NaN},
                        'Premature_Partial': {'BcSpot': True, 'MescSpot': np.NaN, 'BloodPresent': np.NaN,
                                              'MescProblem': True, 'Error': '', 'Version': np.NaN}
                        }

    # Construct expected_results and path constants
    image_path = []
    expected_result = pd.DataFrame(columns=category_results['Good'].keys())
    for cf in category_folders:
        data_path = os.path.join(main_path, cf)
        cf_image_paths = [os.path.join(data_path, f) for f in os.listdir(data_path)]
        image_path += cf_image_paths
        cf_expected_result = pd.DataFrame(category_results[cf], index=range(len(cf_image_paths)))
        expected_result = expected_result.append(cf_expected_result, ignore_index=True)

    return expected_result, image_path


def load_sorted_results_table():
    """
    Use integer code folders (0, 1, 2, etc.) with sorted samples to construct expected result and image paths folder
    list.

    :return: Tuple with pandas data frame with expected/correct result and list of image folders
    """

    main_path = '/media/anders/-Anders-5-/BluSense/ImageLibrary_Plasma_Sorted'
    category_folders = [cf for cf in os.listdir(main_path) if cf.isdigit() and len(cf) == 1]

    image_path = []
    expected_result = pd.DataFrame(columns=integer_to_result_dict(0).keys())
    for cf in category_folders:
        data_path = os.path.join(main_path, cf)
        cf_image_paths = [os.path.join(data_path, f) for f in os.listdir(data_path)]
        image_path += cf_image_paths
        cf_expected_result = pd.DataFrame(integer_to_result_dict(int(cf)), index=range(len(cf_image_paths)))
        expected_result = expected_result.append(cf_expected_result, ignore_index=True)

    return expected_result, image_path


def integer_to_result_dict(result_integer, blood_test=False):
    # Set up default result, i.e. exit code 0
    """
    Convert integer/exit code to result dictionary as returned AllChecks.py: sanity_checker
    """
    result = {'Error': '', 'BcSpot': True, 'MescSpot': True, 'MescProblem': False,
              'BloodPresent': True if blood_test else np.NaN, 'Version': np.NaN}

    # If exit code -1, some error occurred. Error text can vary.
    if result_integer == -1:
        result['Error'] = 'Error'

    # Change default dict based on binary code
    change_key = ['BloodPresent', 'MescProblem', 'MescSpot', 'BcSpot']
    result_binary = '{0:04b}'.format(result_integer)
    for index, key in enumerate(change_key):
        if result_binary[index] == '1':
            result[key] = not result[key]

    # If MescProblem has been changed, also change MescSpot
    if result['MescProblem']:
        result['MescSpot'] = np.NaN

    return result


if __name__ == "__main__":

    # Get expected result
    expected_result, image_path = load_expected_result_table()
    # expected_result, image_path = load_sorted_results_table()

    # Initialize comparison DataFrame
    compare_result = pd.DataFrame(np.nan, index=range(len(image_path)), columns=expected_result.columns)

    # Run test and compare actual result to expectation
    start_time = time.time()
    for i in range(len(image_path)):
        image_files = [os.path.join(image_path[i], f) for f in os.listdir(image_path[i])]
        blood_test = not np.isnan(expected_result['BloodPresent'][i])
        actual_result = qc.sanity_checker(image_files, blood_test)
        actual_result['MescProblem'] = actual_result['MescProblem'] > 0

        for c in actual_result.keys():
            compare_result[c][i] = expected_result[c][i] == actual_result[c]
        print 'Calculating on folder: ' + str(i + 1) + '/' + str(len(image_path)) + '...'
    elapsed_time = time.time() - start_time
    print 'Calculation done after {:.0f} sec. Average = {:3.1f} sec.\n'.format(elapsed_time,
                                                                               elapsed_time / len(image_path))

    # Copy MescPremature and MescBubble result from MescProblem, as this is logged under the same entry.
    for table in [expected_result, compare_result]:
        table['MescBubble'] = table['MescProblem']
        table['MescPremature'] = table['MescProblem']

    # Calculate and print false positive/negative error rates
    positive_error = {'BcSpot': False, 'MescSpot': False, 'BloodPresent': False, 'MescProblem': True,
                      'MescBubble': True, 'MescPremature': True}

    for c in positive_error.keys():
        positives = expected_result[c] == positive_error[c]
        negatives = expected_result[c] == (not positive_error[c])
        positives, negatives = (a[~np.isnan(a)] for a in [positives, negatives])
        false_negative_rate = 1 - np.mean(compare_result[c][positives])
        false_positive_rate = 1 - np.mean(compare_result[c][negatives])
        print '{:>13} | False negatives= {:4.1f} % | False positives= {:4.1f} %'.format(c, 100 * false_negative_rate,
                                                                                        100 * false_positive_rate)
