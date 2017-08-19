import AllChecks as qc
import numpy as np
from skimage import measure, data
import os
import matplotlib.pyplot as plt  # To help making debugging easier


def test_bwareafilt():
    """
    Test function for bwareafilt
    """
    # Define list to store expected and actual result
    expected_area = [np.NaN]*5
    expected_mask = [np.NaN]*5
    actual_mask = [np.NaN]*5
    actual_area = [np.NaN]*5

    # Define test image
    test_image = np.array([[1, 1, 0, 1, 1], [1, 0, 0, 0, 1], [1, 0, 1, 0, 1], [1, 0, 0, 0, 0], [0, 0, 1, 1, 0]],
                          dtype='bool')
    label_image = measure.label(test_image.astype('uint8'), background=0)

    # Test  0. Default behavior
    expected_mask[0] = label_image == 1
    expected_area[0] = np.sum(label_image == 1)
    actual_mask[0], actual_area[0] = qc.bwareafilt(test_image)

    # Test 1. Keep zero objects
    expected_mask[1] = np.zeros(np.shape(label_image), dtype='bool')
    expected_area[1] = 0
    actual_mask[1], actual_area[1] = qc.bwareafilt(test_image, 0)

    # Test 2. Keep two largest objects
    expected_mask[2] = np.logical_or(label_image == 1, label_image == 2)
    expected_area[2] = np.array([5, 4])
    actual_mask[2], actual_area[2] = qc.bwareafilt(test_image, 2)

    # Test 3. Keep too many  largest objects
    expected_mask[3] = test_image
    expected_area[3] = np.array([5, 4, 2, 1])
    actual_mask[3], actual_area[3] = qc.bwareafilt(test_image, n=10)

    # Test 4. Only keep objects with area 2 - 4
    expected_mask[4] = np.logical_or(label_image == 2, label_image == 4)
    expected_area[4] = np.array([4, 2])
    actual_mask[4], actual_area[4] = qc.bwareafilt(test_image, n=10, area_range=[2, 4])

    # Run comparison
    for i in range(len(actual_mask)):
        assert np.all(actual_mask[i] == expected_mask[i])
        assert np.all(actual_area[i] == expected_area[i])


def test_corr2d():
    """
    Test function for corr2d
    """

    # Define list to store expected and actual result
    expected_transition = [np.NaN]*5
    actual_transition = [np.NaN]*5

    # Load test image and crop at different locations
    full_image = data.coins()
    crop_0 = full_image[10: 90, 300:380]
    crop_1 = full_image[20:100, 300:380]
    crop_2 = full_image[10: 90, 290:370]
    crop_3 = full_image[00: 80, 280:360]
    crop_4 = full_image[11: 90, 301:380]

    # Test 0. No movement for images with even side length
    expected_transition[0] = np.array([0, 0])
    actual_transition[0] = qc.corr2d(crop_0, crop_0)

    # Test 1. No movement for images with odd side length
    expected_transition[1] = np.array([0, 0])
    actual_transition[1] = qc.corr2d(crop_4, crop_4)

    # Test 2. Movement along axis 1
    expected_transition[2] = np.array([10, 0])
    actual_transition[2] = qc.corr2d(crop_0, crop_1)

    # Test 3. Movement along axis 2
    expected_transition[3] = np.array([0, -10])
    actual_transition[3] = qc.corr2d(crop_0, crop_2)

    # Test 4. Movement along both axis, above max movement
    expected_transition[4] = np.array([-10, -20])
    actual_transition[4] = qc.corr2d(crop_0, crop_3, max_movement=25)

    # Run comparison
    assert np.all(np.abs(expected_transition - np.array(actual_transition)) < 1)


def test_get_overlap_images():
    """
    Test function for get_overlap_images
    """

    # Define list to store expected and actual result
    expected_images = [np.NaN]*4
    actual_images = [np.NaN]*4

    # Make test images
    j, i = np.meshgrid(np.arange(0, 6), np.arange(0, 6))
    img1 = i + j
    img2 = -img1

    # Test 0. Images with same size
    expected_images[0] = (img1, img2)
    actual_images[0] = qc.get_overlap_images(img1, img2)

    # Test 1. Reduced size of image 2
    expected_images[1] = (img1[1:5, 1:5], img2[:4, :4])
    actual_images[1] = qc.get_overlap_images(img1, img2[:4, :4])

    # Test 2. Include translation
    expected_images[2] = (img1[1:, :5], img2[:5, 1:])
    actual_images[2] = qc.get_overlap_images(img1, img2, (1, -1))

    # Test 3. Make translation larger than images
    expected_images[3] = (np.zeros((0, 6), 'int'), np.zeros((0, 6), 'int'))
    actual_images[3] = qc.get_overlap_images(img1, img2, (10, 0))

    # Run comparison
    for i in range(len(actual_images)):
        for j in range(1):
            assert np.all(expected_images[i][j] == actual_images[i][j])


def test_imgradient():
    """
    Test function for imgradient
    """

    # Define list to store expected and actual result
    expected_images = [np.NaN]*4
    actual_images = [np.NaN]*4

    # Make test image
    j, i = np.meshgrid(np.arange(0, 8), np.arange(0, 8))
    test_image = np.logical_or(i == 2, i == 5).astype('uint8')

    # Test 0. Image with constant 170 value
    expected_images[0] = np.zeros((10, 10), 'uint8')
    actual_images[0] = qc.imgradient(170*np.ones((10, 10), 'uint8'))

    # Test 1. Image with horizontal bars
    expected_images[1] = 4*np.isin(i, [1, 3, 4, 6])
    actual_images[1] = qc.imgradient(test_image)

    # Test 2. Image with vertical bars
    expected_images[2] = expected_images[1].T
    actual_images[2] = qc.imgradient(test_image.T)

    # Test 3. Image with grid
    expected_images[3] = np.sqrt(expected_images[1]**2 + expected_images[2]**2)
    actual_images[3] = qc.imgradient(test_image + test_image.T)

    # Run comparison
    for i in range(len(actual_images)):
        assert np.all(expected_images[i] == actual_images[i])


def test_imreconstruct():
    """
    Test function for imreconstruct
    """

    # Define list to store expected and actual result
    expected_mask = [np.NaN] * 3
    actual_mask = [np.NaN] * 3

    test_mask = np.array([[1, 1, 0, 1, 1], [1, 0, 0, 0, 1], [1, 0, 1, 0, 1], [1, 0, 0, 0, 0], [0, 0, 1, 1, 0]],
                         dtype='bool')
    label_image = measure.label(test_mask.astype('uint8'), background=0)

    # Test 0. Mask with zero pixels
    marker = np.zeros((5, 5), dtype='bool')
    marker[-1, -1] = True
    expected_mask[0] = np.zeros((5, 5), dtype='bool')
    actual_mask[0] = qc.imreconstruct(marker, test_mask)

    # Test 1. Mask with one pixel
    marker = np.zeros((5, 5), dtype='bool')
    marker[0, 0] = True
    expected_mask[1] = label_image == 1
    actual_mask[1] = qc.imreconstruct(marker, test_mask)

    # Test 2. Marker with filled inner region
    marker = np.zeros((5, 5), dtype='bool')
    marker[1:3, 1:3] = True
    expected_mask[2] = label_image == 3
    actual_mask[2] = qc.imreconstruct(marker, test_mask)

    # Run comparison
    for i in range(len(actual_mask)):
        assert np.all(expected_mask[i] == actual_mask[i])


def test_maxspan():
    """
    Test function for maxspan
    """

    # Define list to store expected and actual result
    expected_length = [np.NaN] * 3
    actual_length = [np.NaN] * 3

    # Test 0. No True
    expected_length[0] = 0
    actual_length[0] = qc.max_span(np.zeros(10, dtype=bool))

    # Test 1. Some Trues
    expected_length[1] = 4
    actual_length[1] = qc.max_span(np.array([0, 1, 0, 1, 1], dtype=bool))

    # Test 2.  One Trues
    expected_length[2] = 1
    actual_length[2] = qc.max_span(np.array([1, 0, 0, 0, 0], dtype=bool))

    # Run comparison
    for i in range(len(actual_length)):
        assert expected_length[i] == actual_length[i]


if __name__ == "__main__":
    # os.system('python -m pytest -v QcUnitTest.py') # This will replace line below if no virtual environment
    os.system('/home/anders/anaconda3/envs/cv/bin/python -m pytest -v QcUnitTest.py')

    # To run individual tests, comment lines above and uncomment lines below here
    # test_bwareafilt()
    # test_corr2d()
    # test_get_overlap_images()
    # test_imgradient()
    # test_imreconstruct()
    # test_maxspan()


# How to run, e.g. in terminal in PyCharm:
# $ source activate cv
# $ pytest QcUnitTest.py
