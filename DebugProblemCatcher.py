import AllChecks as qc
import QcPerformanceTest as qct
import numpy as np
import pandas as pd
import os
import pdb

# Create list of tests that the debugger will test. The list entries must be the keys names in the result dict.
entries_to_test = ['MescProblem']

# Load table of results
expected_result, image_path = qct.load_expected_result_table()

# Run test and compare actual result to expectation
for i in range(len(image_path)): #TODO this line should be: range(len(image_path))

    result = {'Error': '', 'BcSpot': None, 'MescSpot': None, 'MescProblem': None, 'BloodPresent': None}
    image_files = [os.path.join(image_path[i], f) for f in os.listdir(image_path[i])]
    dir_link = '<a href=' + image_path[i] + '>image_dir</a>'

    # Image 0
    image_idx = 0
    reference_chambers, _ = qc.find_chambers(image_files[image_idx])

    # Image 1
    image_idx = 1
    blood_test = not np.isnan(expected_result['BloodPresent'][i])
    chambers, result['BloodPresent'] = qc.find_chambers(image_files[image_idx], blood_test)
    if blood_test and np.isin('BloodPresent', entries_to_test) and result['BloodPresent'] != expected_result['BloodPresent'][i]:
        print 'BloodPresent failed: ' + dir_link

    result['BcSpot'] = qc.detect_spot_mesc(chambers[0], reference_chambers[0]) # TODO replace this with real function
    if np.isin('BcSpot', entries_to_test) and result['BcSpot'] != expected_result['BcSpot'][i]:
        print 'BcSpot failed: ' + dir_link

    result['MescProblem'] = qc.detect_badfill_mesc(chambers[1], reference_chambers[1])
    dry_mesc_chamber = chambers[1]

    # Image 2
    image_idx = 2
    chambers, _ = qc.find_chambers(image_files[image_idx])
    result['MescSpot'] = qc.detect_spot_mesc(chambers[1], dry_mesc_chamber)
    if np.isin('MescSpot', entries_to_test) and result['MescSpot'] != expected_result['MescSpot'][i]:
        print 'MescSpot failed: ' + dir_link

    if result['MescProblem'] == 0:
        result['MescProblem'] = qc.detect_badfill_mesc(chambers[1], reference_chambers[1])
    if np.isin('MescProblem', entries_to_test) and (result['MescProblem'] > 0) != expected_result['MescProblem'][i]:
        print 'MescProblem failed: ' + dir_link
        temp = 'set breakpoint here'

    print 'Calculating on folder: ' + str(i + 1) + '/' + str(len(image_path)) + '...'

print('DebugProblemCatcher done.')