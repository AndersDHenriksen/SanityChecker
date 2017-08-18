import AllChecks as qc
import numpy as np
import pandas as pd
import os


def load_expected_result_table():
    """
    Use excel table with manual annotation of images to construct expected result.

    :return: Dataframe with expected/correct result
    """

    # Load truth table
    data_path = r'C:\Users\310229518\Google Drive\BluSense'
    # data_path = '/media/anders/-Anders-3-/Google Drev/BluSense'
    truth_table = pd.read_excel(os.path.join(data_path, 'ImageLibrary_Plasma', 'ImageLibrary_Plasma_TruthTable.xlsx'))
    truth_table.drop(np.argwhere(truth_table['Blacklist']>0), inplace=True)

    # Create list of paths
    image_path = [os.path.join(data_path, truth_table['MainPath'][i], str(truth_table['Folder'][i])).replace('\\',
                  os.sep) for i in range(truth_table.shape[0])]

    # Create expected result. Note this must have same columns as qc.sanity_checker output keys
    expected_result = pd.DataFrame()
    expected_result[['BcSpot', 'MescSpot', 'BloodPresent']] = truth_table[['BcSpot', 'MescSpot', 'Blood']]
    expected_result.loc[truth_table['Leaking'] > 0, ['BcSpot', 'MescSpot']] = 0
    expected_result['MescProblem'] = np.logical_or(truth_table['Mesc Premature'], truth_table['Mesc Bubble'])
    expected_result['Error'] = ''

    return expected_result, image_path

if __name__ == "__main__":

    # Get expected result
    expected_result, image_path = load_expected_result_table()

    # Initialize comparison DataFrame
    compare_result = pd.DataFrame(np.nan, index=range(len(image_path)), columns=expected_result.columns)

    # Run test and compare actual result to expectation
    for i in range(len(image_path)):
        image_files = [os.path.join(image_path[i], f) for f in os.listdir(image_path[i])]
        blood_test = not np.isnan(expected_result['BloodPresent'][i])
        actual_result = qc.sanity_checker(image_files, blood_test)
        actual_result['MescProblem'] = actual_result['MescProblem'] > 0

        for c in actual_result.keys():
            compare_result[c][i] = expected_result[c][i] == actual_result[c]
        print 'Calculating on folder: ' + str(i + 1) + '/' + str(len(image_path)) + '...'

    # Calculate and print false positive/negative error rates
    positive_error = {'BcSpot': False, 'MescSpot': False, 'BloodPresent': False, 'MescProblem': True}
    for c in expected_result.columns[:-1]:
        positives = expected_result[c] == positive_error[c]
        negatives = expected_result[c] == (not positive_error[c])
        positive_error_rate = 1 - np.mean(compare_result[c][positives])
        negative_error_rate = 1 - np.mean(compare_result[c][negatives])
        print '{:>12} | False positives= {:4.1f} % | False negatives= {:4.1f} %'.format(c, 100 * positive_error_rate,
                                                                                        100 * negative_error_rate)