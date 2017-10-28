import AllChecks as qc
import QcPerformanceTest as qct
import numpy as np
import os

# Create list of tests that the debugger will test. The list entries must be the keys names in the result dict.
# Remember to also add breakpoints in gutter, look for '_ = set breakpoint here'
entries_to_test = ['BloodPresent', 'BcSpot', 'MescSpot', 'MescProblem']

# Load table of results, choose one of the lines below
expected_result, image_path = qct.load_expected_result_table()
# expected_result, image_path = qct.load_sorted_results_table()

# Run test and compare actual result to expectation
for i in range(0, len(image_path)):
    print 'Calculating on folder: ' + str(i) + '/' + str(len(image_path)) + '...'

    result = {'Error': '', 'BcSpot': None, 'MescSpot': None, 'MescProblem': None, 'BloodPresent': None}
    image_files = [os.path.join(image_path[i], f) for f in os.listdir(image_path[i])]
    dir_link = '<a href=' + image_path[i] + '>image_dir</a>'

    # Image 0
    image_idx = 0
    reference_chambers, _, _ = qc.find_chambers(image_files[image_idx])

    # Image 1
    image_idx = 1
    blood_test = not np.isnan(expected_result['BloodPresent'][i])
    chambers, _, result['BloodPresent'] = qc.find_chambers(image_files[image_idx], blood_test)
    if blood_test and np.isin('BloodPresent', entries_to_test) and result['BloodPresent'] != \
            expected_result['BloodPresent'][i]:
        print 'BloodPresent failed: ' + dir_link
        _ = 'set breakpoint here'

    result['BcSpot'] = qc.detect_spot_bc(chambers[0], reference_chambers[0])
    if np.isin('BcSpot', entries_to_test) and result['BcSpot'] != expected_result['BcSpot'][i]:
        print 'BcSpot failed: ' + dir_link
        _ = 'set breakpoint here'

    result['MescProblem'] = qc.detect_badfill_mesc(chambers[1], reference_chambers[1])
    dry_mesc_chamber = chambers[1]

    # Image 2
    image_idx = 2
    chambers, _, _ = qc.find_chambers(image_files[image_idx])
    result['MescSpot'] = qc.detect_spot_mesc(chambers[1], dry_mesc_chamber)
    if np.isin('MescSpot', entries_to_test) and result['MescSpot'] != expected_result['MescSpot'][i] and not np.isnan(
            expected_result['MescSpot'][i]):
        print 'MescSpot failed: ' + dir_link
        _ = 'set breakpoint here'
        qc.detect_spot_mesc(chambers[1], dry_mesc_chamber, True)

    if result['MescProblem'] == 0:
        result['MescProblem'] = qc.detect_badfill_mesc(chambers[1], reference_chambers[1])
    if np.isin('MescProblem', entries_to_test) and (result['MescProblem'] > 0) != expected_result['MescProblem'][i]:
        print 'MescProblem failed: ' + dir_link
        _ = 'set breakpoint here'

print('DebugProblemCatcher done.')
