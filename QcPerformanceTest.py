import AllChecks as qc
import numpy as np
import pandas as pd
import os

#if __name__ == "__main__":

# Load truth table
data_path = r'C:\Users\310229518\Google Drive\BluSense'
truth_table = pd.read_excel(data_path + r'\ImageLibrary_Plasma\ImageLibrary_Plasma_TruthTable.xlsx')
truth_table.drop(np.argwhere(truth_table['Blacklist']>0), inplace=True)

#Create list of paths
image_path = [data_path +'\\' + truth_table['MainPath'][i] + str(truth_table['Folder'][i]) for i in
              range(truth_table.shape[0])]

# Create expected result. Note this must have same columns as qc.sanity_checker output keys
expected_result = pd.DataFrame()
expected_result[['BcSpot', 'MescSpot', 'BloodPresent']] = truth_table[['BcSpot', 'MescSpot', 'Blood']]
expected_result.loc[truth_table['Leaking'] > 0, ['BcSpot', 'MescSpot']] = 0
expected_result['MescProblem'] = np.logical_or(truth_table['Mesc Premature'], truth_table['Mesc Bubble'])
expected_result['Error'] = ''

# Initialize comparison dataframe
compare_result = pd.DataFrame(np.nan, index=range(len(image_path)), columns=expected_result.columns)

# Run test and compare actual result to expectation
for i in range(len(image_path)):
    print 'Calculating on folder: ' + str(i+1) + '/' + str(len(image_path)) + '...'
    image_files = [image_path[i] + '\\' + f for f in os.listdir(image_path[i])]
    blood_test = not np.isnan(expected_result['BloodPresent'][i])
    actual_result = qc.sanity_checker(image_files, blood_test)
    actual_result['MescProblem'] = actual_result['MescProblem'] > 0

    for c in actual_result.keys():
        compare_result[c][i] = expected_result[c][i] == actual_result[c]
        # if type(expected_result[c][i]) == np.float64 and np.isnan(expected_result[c][i]):
        #     compare_result[c][i] = np.nan

# Calculate and print false positive/negative error rates
positive_error = {'BcSpot': False, 'MescSpot': False, 'BloodPresent': False, 'MescProblem': True}
for c in expected_result.columns[:-1]:
    positives = expected_result[c] == positive_error[c]
    negatives = expected_result[c] == (not positive_error[c])
    positive_error_rate = 1 - np.mean(compare_result[c][positives])
    negative_error_rate = 1 - np.mean(compare_result[c][negatives])
    print c + '| False positives= ' + str(100*positive_error_rate) + ' % | False negatives=  ' + str(100*positive_error_rate) + ' %'