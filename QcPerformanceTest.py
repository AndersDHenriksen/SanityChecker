import AllChecks as qc
import numpy as np
import pandas as pd
import os
import time


def load_expected_result_table():
    """
    Use excel table with manual annotation of images to construct expected result and image paths folder list.

    :return: Tuple with pandas data frame with expected/correct result and list of image folders
    """

    # Define main paths, categories and their results
    main_path = '/media/anders/-Anders-3-/Google Drev/BluSense/ImageLibrary_Plasma'
    category_folders = ['Good', 'No_Beads', 'No_Reagents', 'No_Sample', 'Not_Filled', 'Premature_Partial',
                        'Premature_Full']
    category_results = {'Good': {'BcSpot': True, 'MescSpot': True, 'BloodPresent': np.NaN, 'MescProblem': False,
                                 'Error': ''},
                        'No_Beads': {'BcSpot': True, 'MescSpot': False, 'BloodPresent': np.NaN, 'MescProblem': False,
                                     'Error': ''},
                        'No_Reagents': {'BcSpot': False, 'MescSpot': True, 'BloodPresent': np.NaN, 'MescProblem': False,
                                        'Error': ''},
                        'No_Sample': {'BcSpot': False, 'MescSpot': False, 'BloodPresent': np.NaN, 'MescProblem': False,
                                      'Error': ''},
                        'Not_Filled': {'BcSpot': True, 'MescSpot': True, 'BloodPresent': np.NaN, 'MescProblem': True,
                                       'Error': ''},
                        'Premature_Full': {'BcSpot': True, 'MescSpot': False, 'BloodPresent': np.NaN,
                                           'MescProblem': False, 'Error': ''},
                        'Premature_Partial': {'BcSpot': True, 'MescSpot': True, 'BloodPresent': np.NaN,
                                              'MescProblem': True, 'Error': ''}
                        }

    # Construct expected_results and path constants
    image_path = []
    expected_result = pd.DataFrame(columns=category_results['Good'].keys())
    for cf in category_folders:
        data_path = os.path.join(main_path, cf)
        cf_image_paths = [os.path.join(data_path, f) for f in os.listdir(data_path)]
        image_path = image_path + cf_image_paths
        cf_expected_result = pd.DataFrame(category_results[cf], index=range(len(cf_image_paths)))
        expected_result = expected_result.append(cf_expected_result, ignore_index=True)

    return expected_result, image_path


if __name__ == "__main__":

    # Get expected result
    expected_result, image_path = load_expected_result_table()

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
        false_negative_rate = 1 - np.mean(compare_result[c][positives])
        false_positive_rate = 1 - np.mean(compare_result[c][negatives])
        print '{:>13} | False negatives= {:4.1f} % | False positives= {:4.1f} %'.format(c, 100 * false_negative_rate,
                                                                                        100 * false_positive_rate)
