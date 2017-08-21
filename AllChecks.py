import numpy as np
import cv2
from skimage import morphology, measure, segmentation
import astropy.convolution
import argparse
import os
# packages below this line are not crucial
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import glob


def find_chambers(image_path, blood_also=False, debug=False):
    """ Find chambers in camera image

    Chambers are identified using rough idea of their location and using cross-correlation to a reference image.

    :param image_path: String with path and name of image.
    :param blood_also: Bool, if blood should be looked for in BSS and OFC2.
    :param debug: Bool, if plot with chambers should be produced
    :return: 3-item tuple with List of 2 chamber objects, possible error string, bool if blood was detected.
    """

    # Define settings and output
    setting = {'PositionExpected': (1380, 2500), 'CutSide': 500, 'BcOffsetToRefImg': (-34, -610), 'BcRi': 146,
               'BcRj': 160, 'MescOffsetToRefImg': (2, 24), 'MescR': 142, 'BloodRatio': (0.25, 0.5), 'xCorMin': 0.1}
    poly_bss = np.array(
        [(2251, 1968), (2194, 2282), (2233, 2408), (2308, 2426), (2418, 1866), (2359, 1857), (2251, 1968)], dtype='int')
    poly_ofc2 = np.array([(560, 2480), (524, 2109), (374, 2381), (560, 2480)], dtype='int')
    chambers = [None]*2
    error = ''
    blood_present = [None] * 2

    # Load images and reference image
    image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2GRAY)
    script_path = os.path.dirname(os.path.abspath(__file__))
    image_mesc_chamber = cv2.cvtColor(cv2.imread(os.path.join(script_path, 'Data', 'labelfree_mesc_chamber.png')),
                                      cv2.COLOR_BGR2GRAY)
    mask_mesc_chamber = imgradient(image_mesc_chamber) > 60

    cut_out = image[setting['PositionExpected'][0] - setting['CutSide']:setting['PositionExpected'][0] + setting['CutSide'],
                    setting['PositionExpected'][1] - setting['CutSide']:setting['PositionExpected'][1] + setting['CutSide']]

    xcor_mesc = astropy.convolution.convolve_fft(imgradient(cut_out) > 60, mask_mesc_chamber[::-1, ::-1], 'wrap')
    if np.max(xcor_mesc) < setting['xCorMin']:
        error = 'Chambers could not be detected for:' + image_path
        return chambers, error, blood_present
    i_max, j_max = np.unravel_index(xcor_mesc.argmax(), xcor_mesc.shape)
    d_xcor = np.array([i_max, j_max]) - (np.array(np.shape(xcor_mesc)) + 1)/2
    idx_ref = np.array(setting['PositionExpected']) + d_xcor

    cor_bc = idx_ref + np.array(setting['BcOffsetToRefImg'])
    chambers[0] = Chamber('BC', image,  cor_bc[0], cor_bc[1], setting['BcRi'], setting['BcRj'])

    cor_mesc = idx_ref + np.array(setting['MescOffsetToRefImg'])
    chambers[1] = Chamber('MESC', image, cor_mesc[0], cor_mesc[1], setting['MescR'], setting['MescR'])

    # Detect blood, by looking at intensity in ROI polys
    if blood_also:
        image_mean = np.mean(image)
        # Loop through the two polygon defining where to look for blood, and calc. intensity-ratio to image.
        for i, poly in enumerate([poly_bss, poly_ofc2]):
            mask = np.zeros(image.shape, dtype='uint8')
            poly += d_xcor[::-1]
            cv2.fillConvexPoly(mask, poly, True)
            blood_present[i] = np.mean(image[mask.astype('bool')]) / image_mean < setting['BloodRatio'][i]

    # Show image output if true debug flag
    if debug:
        plt.figure()
        plt.imshow(image)
        ax = plt.gca()
        ellipse0 = Ellipse(xy=(cor_bc[::-1]), width=2 * setting['BcRj'], height=2 * setting['BcRi'], edgecolor='r',
                           fc='None', lw=2)
        ellipse1 = Ellipse(xy=(cor_mesc[::-1]), width=2 * setting['MescR'], height=2 * setting['MescR'], edgecolor='g',
                           fc='None', lw=2)
        poly0 = plt.Polygon(poly_bss, ec='b', fc='none')
        poly1 = plt.Polygon(poly_ofc2, ec='y', fc='none')
        [ax.add_patch(p) for p in [ellipse0, ellipse1, poly0, poly1]]

    return chambers, error, np.all(blood_present)


def imreconstruct(marker, mask):
    """Performs morphological reconstruction of the image marker under the image mask."""
    return morphology.reconstruction(np.logical_and(marker, mask), mask).astype('bool')


def imgradient(img):
    """ Calculates the (Sobel) gradient magnitude of the image."""
    return np.sqrt(cv2.Sobel(img, cv2.CV_64F, 1, 0) ** 2 + cv2.Sobel(img, cv2.CV_64F, 0, 1) ** 2)


def bwareafilt(mask, n=1, area_range=(0, np.inf)):
    """Extract objects from binary image by size """
    labels = measure.label(mask.astype('uint8'), background=0)
    area_idx = np.arange(1, np.max(labels) + 1)
    areas = np.array([np.sum(labels == i) for i in area_idx])
    inside_range_idx = np.logical_and(areas >= area_range[0], areas <= area_range[1])
    area_idx = area_idx[inside_range_idx]
    areas = areas[inside_range_idx]
    keep_idx = area_idx[np.argsort(areas)[::-1][0:n]]
    kept_areas = areas[np.argsort(areas)[::-1][0:n]]
    if np.size(kept_areas) == 0:
        kept_areas = np.array([0])
    if n == 1:
        kept_areas = kept_areas[0]
    kept_mask = np.isin(labels, keep_idx)
    return kept_mask, kept_areas


def max_span(a):
    """ Return the lenght/span of true elements in input. """
    true_idx = np.argwhere(a)
    if np.size(true_idx) == 0:
        return 0
    return true_idx[-1][0] - true_idx[0][0] + 1


def detect_spot_bc(chamber, reference_chamber, debug=False):
    """ Detect reagent spot in BC chamber.

    Look for a reagent spot using the intensity difference between the initial chamber
    and the plasma filled chamber.

    :param chamber: BC chamber where spot should have disappeared.
    :param reference_chamber: BC chamber to detect spot in.
    :param debug: Bool, if True visual plot to understand algorithm steps.
    :return: Bool, if True reagent spot was detected.
    """

    setting = {'CompensateAverage': True, 'RatioThres': 0.45, 'UseMedianFilter': True}
    conclusions = ['Spot NOT detected in BC chamber.', 'Spot detected in BC chamber.']

    # Align chamber images and radius image
    chm_img, ref_img = get_overlap_images(chamber.Img, reference_chamber.Img)
    translation = corr2d(chm_img, ref_img)
    r, _ = get_overlap_images(chamber.R, reference_chamber.R)
    r, _ = get_overlap_images(r, ref_img, translation)
    chm_img, ref_img = get_overlap_images(chm_img, ref_img, translation)

    half_point = int(chm_img.shape[0]/2)
    chm_img = chm_img[:half_point, :]
    ref_img = ref_img[:half_point, :]
    r = r[:half_point, :]

    # Perform filtering to reduce noise and level out mean offset
    if setting['UseMedianFilter']:
        chm_img = cv2.medianBlur(chm_img, 3)
        ref_img = cv2.medianBlur(ref_img, 3)

    if setting['CompensateAverage']:
        mask = np.logical_and(r < .9, r > .5)
        compensation = (np.mean(chm_img[mask]) - np.mean(ref_img[mask])).clip(min=-2)
    else:
        compensation = 0

    # Get difference between chambers and detect beads
    diff_img = (ref_img.astype('int16')+compensation - chm_img).clip(min=0)
    ratio = np.mean(diff_img[r < .75] > 4)
    has_spot = ratio > setting['RatioThres']

    if debug:
        plt.figure()
        plt.subplot(2, 2, 1)
        plt.imshow(diff_img)
        plt.title(conclusions[has_spot])
        plt.subplot(2, 2, 2)
        plt.imshow(np.logical_and(diff_img > 4, r < .75))
        plt.title('Ratio: %.2f' % ratio)
        plt.subplot(2, 2, 3)
        plt.imshow(ref_img)
        plt.subplot(2, 2, 4)
        plt.imshow(chm_img)
    return has_spot


def detect_spot_mesc(chamber, reference_chamber, debug=False):
    """ Detect beads spot in MESC chamber.

    Look for a magnetic beads spot using the intensity difference between the initial chamber
    and the plasma filled chamber.

    :param chamber: MESC chamber where spot should have disappeared.
    :param reference_chamber: MESC chamber to detect spot in.
    :param debug: Bool, if True visual plot to understand algorithm steps.
    :return: Bool, if True beads spot was detected.
    """

    setting = {'CompensateAverage': True, 'RatioThres': 0.14, 'UseLinearCorrection': True, 'UseRadialCorrection': True,
               'UseMedianFilter': True}
    conclusions = ['Beads NOT detected in MESC chamber.', 'Beads detected in MESC chamber.']

    # First try simple detection
    has_beads = simple_bead_spot_detection(reference_chamber)
    if has_beads:
        return has_beads - simple_bead_spot_detection(chamber)

    # Compensate out intensity slopes
    if setting['UseLinearCorrection']:
        linear_correction(chamber)
        linear_correction(reference_chamber)

    # Compensate out radial intensities
    if setting['UseRadialCorrection']:
        radial_correction(chamber)
        radial_correction(reference_chamber)

    # Align chamber images and radius image
    chm_img, ref_img = get_overlap_images(chamber.Img, reference_chamber.Img)
    translation = corr2d(chm_img, ref_img)
    r, _ = get_overlap_images(chamber.R, reference_chamber.R)
    r, _ = get_overlap_images(r, ref_img, translation)
    chm_img, ref_img = get_overlap_images(chm_img, ref_img, translation)

    # Perform filtering to reduce noise and level out mean offset
    if setting['UseMedianFilter']:
        chm_img = cv2.medianBlur(chm_img, 3)
        ref_img = cv2.medianBlur(ref_img, 3)

    if setting['CompensateAverage']:
        mask = np.logical_and(r < .9, r > .5)
        compensation = (np.mean(chm_img[mask]) - np.mean(ref_img[mask])).clip(min=-2)
    else:
        compensation = 0

    # Get difference between chambers and detect beads
    diff_img = (ref_img.astype('int16')+compensation - chm_img).clip(min=0)
    ratio = np.mean(diff_img[r < .75] > 4)
    has_beads = ratio > setting['RatioThres']

    # Also try to find centroid of largest object. Go to centroid calc ratio in vicinity
    if not has_beads and ratio > .1:
        mask = np.logical_and(diff_img > 4, r < .75)
        j, i = np.meshgrid(np.arange(1, np.size(mask, 1) + 1), np.arange(1, np.size(mask, 0) + 1))
        center_mask = np.logical_and((i - np.mean(i[mask]))**2 + (j - np.mean(j[mask]))**2 < 70**2, r < .75)
        new_ratio = np.sum(mask[center_mask]).astype('f') / np.sum(center_mask)
        has_beads = new_ratio > 2*setting['RatioThres'] and np.mean(j[mask])/mask.shape[1] < .7

    if debug:
        plt.figure()
        plt.subplot(2, 2, 1)
        plt.imshow(diff_img)
        plt.title(conclusions[has_beads])
        plt.subplot(2, 2, 2)
        plt.imshow(np.logical_and(diff_img > 4, r < .75))
        plt.title('Ratio: %.2f' % ratio)
        plt.subplot(2, 2, 3)
        plt.imshow(ref_img)
        plt.subplot(2, 2, 4)
        plt.imshow(chm_img)
    return has_beads


def simple_bead_spot_detection(chamber):
    """ Try to identify if bead spot is present based on area and solidity for intensity thresholding """

    inner = chamber.R < .75
    spot = [np.NaN]*7
    background = round(np.mean(chamber.Img[chamber.R < .8]))
    kernel_r3 = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    for i in range(7):
        intensity_mask = np.logical_and(chamber.Img > (i + background), inner)
        intensity_mask = segmentation.clear_border(np.logical_or(intensity_mask, np.logical_not(inner)))
        only_biggest, biggest_area = bwareafilt(intensity_mask)
        area_ratio = biggest_area.astype('f') / np.sum(inner)
        clump_ratio = biggest_area.astype('f') / np.sum(intensity_mask)
        eroded_area = np.sum(cv2.erode(only_biggest.astype('uint8'), kernel_r3))
        solidity = eroded_area.astype('f') / biggest_area
        spot[i] = clump_ratio > .75 and area_ratio > .1 and solidity > .4
    return np.any(spot)


def linear_correction(chamber):
    """ Compensate out intensity slopes"""
    inner_mask = chamber.R < .9
    o = np.ones(np.sum(inner_mask))
    coef = np.linalg.lstsq(np.array([o, chamber.X[inner_mask], chamber.Y[inner_mask]]).T, chamber.Img[inner_mask])[0]
    chamber.Img = np.round(chamber.Img - min(0, coef[1]) * chamber.X - min(0, coef[2]) * chamber.Y).astype('uint8')


def radial_correction(chamber):
    """ Compensate out higher intensity in SW corner"""
    light_angle = 3*np.pi/4
    chamber_angle = np.angle(chamber.X + 1j*chamber.Y)
    mask = np.logical_and.reduce((chamber.R < .9, chamber.R > .5, np.cos(chamber_angle + light_angle) > 0))
    o = np.ones(np.sum(mask))
    coef = np.linalg.lstsq(np.array([o, np.cos(chamber_angle[mask] + light_angle)]).T, chamber.Img[mask])[0].clip(min=0)
    chamber.Img = np.round(chamber.Img - (coef[1] * np.cos(chamber_angle + light_angle)
                                          * np.cos(np.pi / 2 * (chamber.R - 1))).clip(min=0)).astype('uint8')


def detect_badfill_mesc(chamber, reference_chamber, debug=False):
    """ Detect MESC problem, either bubble or overflow.

    The image data from chamber and reference chamber is aligned and subtracted to make a diff image.
    In this diff image vertical lines of high intensity (overflow)
    and bubbles whose edge are intensity than the background are both detected.

    :param chamber: MESC chamber object from image 2 or 3.
    :param reference_chamber: MESC chamber object before liquid, i.e. from image 1.
    :param debug: Bool, if True visual plot to understand algorithm steps.
    :return: int error code. 0 = no problems. 1 = premature transfer. 2 = bubble.
    """

    setting = {'RCutOff': 0.95, 'CompensateAverage': True, 'UseLinearCorrection': True, 'UseRadialCorrection': True,
               'UseMedianFilter': True}
    conclusions = ['MESC stayed dry.', 'MESC overflow.', 'MESC bubble.']

    # Compensate out intensity slopes
    if setting['UseLinearCorrection']:
        linear_correction(chamber)
        linear_correction(reference_chamber)

    # Compensate out radial intensities
    if setting['UseRadialCorrection']:
        radial_correction(chamber)
        radial_correction(reference_chamber)

    # Calculate translation between images and overlaps
    chm_img, ref_img = get_overlap_images(chamber.Img, reference_chamber.Img)
    translation = corr2d(chm_img, ref_img)
    r, _ = get_overlap_images(chamber.R, reference_chamber.Img)
    x, _ = get_overlap_images(chamber.X, reference_chamber.Img)
    r, _ = get_overlap_images(r, ref_img, translation)
    x, _ = get_overlap_images(x, ref_img, translation)
    chm_img, ref_img = get_overlap_images(chm_img, ref_img, translation)

    # Perform filtering to reduce noise and level out mean offset
    if setting['UseMedianFilter']:
        chm_img = cv2.medianBlur(chm_img, 3)
        ref_img = cv2.medianBlur(ref_img, 3)

    if setting['CompensateAverage']:
        mask = np.logical_and(r < .9, r > .5)
        compensation = (np.mean(chm_img[mask]) - np.mean(ref_img[mask])).clip(min=-2)
    else:
        compensation = 0

    # Calculate difference between images, and row averages
    diff_img = (chm_img.astype('int16') - compensation - ref_img).clip(min=0)
    diff_img[r > setting['RCutOff']] = 0
    average_diff = np.sum(diff_img, axis=0) / (.1 + np.sum(r <= setting['RCutOff'], axis=0))

    # Overflow is high intensity line and not last pixels
    max_idx = np.argmax(average_diff)
    mesc_overflow = average_diff[max_idx] > 30 and np.sum(average_diff[max_idx:] > 0) > 4

    # Look for mesc bubble at certain region and size
    if not mesc_overflow and np.mean(diff_img[r < .9]) < 9:
        bubble_thres = min(15, 2 * np.mean(diff_img) + 5)
        bubble_mask = np.logical_and.reduce((diff_img < bubble_thres, r < .9, x < .25))
        _, bubble_areas = bwareafilt(bubble_mask, area_range=(400, 20000))
        mesc_overflow = 2 * (np.sum(bubble_areas) > 250)

    # If print output is desired
    # print conclusions[mesc_overflow]

    if debug:
        plt.figure()
        plt.subplot(1, 2, 1)
        plt.imshow(diff_img)
        plt.title(conclusions[mesc_overflow])
        if 'bubble_mask' in locals():
            plt.subplot(1, 2, 2)
            plt.imshow(bubble_mask)

    return mesc_overflow


def get_overlap_images(img1, img2, translation=None):
    """ Get overlap of images from two images.

    Gets image overlap, which is usefull to make sure output images are of equal size.
    Overlap is calculated by cropping highest i,j values not in both images.
    Further, translation is an optimal output which the algorithm understands as a
    to-do translation of image 2, before calculating overlap.

    :param img1: np.array of uint8, image 1.
    :param img2: np.array of uint8, image 2.
    :param translation: Translation to apply to image2 before calculating overlap.
    :return: Overlapping part of image 1; Overlapping part of image 2.
    """

    if translation is None:
        translation = [0, 0]
    translation = np.round(translation).astype('int')

    # Define outputs to prevent changing inputs
    out_img1 = img1.copy()
    out_img2 = img2.copy()

    # Find index overlap for each dimension
    for d in range(2):
        size1 = np.size(img1, d)
        size2 = np.size(img2, d)
        size_move = int(round((size2 - size1) / 2))
        side1_idx = np.intersect1d(range(size1), np.arange(0, size2) - size_move + translation[d], assume_unique=True)
        side2_idx = np.intersect1d(np.arange(0, size1) + size_move - translation[d], range(size2), assume_unique=True)
        if d == 0:
            out_img1 = out_img1[side1_idx, :]
            out_img2 = out_img2[side2_idx, :]
        else:
            out_img1 = out_img1[:, side1_idx]
            out_img2 = out_img2[:, side2_idx]

    return out_img1, out_img2


def corr2d(img1, img2, max_movement=12):
    """ Calculate translation between image 1 and image 2.

    Using fft based cross-corelation the best match between image1 and image2 is found.
    Image 1 and image 2 must be the same size. If not, use get_overlap_images before hand.

    :param img1: np.array of uint8, image 1.
    :param img2: np.array of uint8, image 2.
    :param max_movement: integer, max translation to look for.
    :return: A list of translations to apply to image 2 for it to overlap with image 1.
    """

    # Calculate fft based 2d cross-correlation
    xcorr2d = astropy.convolution.convolve_fft(img1, img2[::-1, ::-1], 'wrap')

    # Calculate image midpoints
    mid_points = (np.array(np.shape(xcorr2d)) - 1) / 2

    # Crop out 25x25 pixels around midpoint of xcorr result
    xcorr2d_crop = xcorr2d[mid_points[0] - max_movement:mid_points[0] + max_movement + 1,
                           mid_points[1] - max_movement:mid_points[1] + max_movement + 1]

    # Get maximum indexes and recalculate to full image coordinates
    i_idx, j_idx = np.unravel_index(xcorr2d_crop.argmax(), xcorr2d_crop.shape)
    i_idx = i_idx + mid_points[0] - max_movement
    j_idx = j_idx + mid_points[1] - max_movement

    n_fit = 3
    delta_ij = np.array([0, 0])
    if i_idx >= n_fit and j_idx >= n_fit and i_idx + n_fit < np.size(xcorr2d, 0) and \
       j_idx + n_fit < np.size(xcorr2d, 1):
        # Fit data to p(0) + p(1)*x + p(2)*y + p(3)*x^2 + p(4)*xy + p(5)*y^2 and get top point
        fit_data = np.log(xcorr2d[i_idx - n_fit:i_idx + n_fit + 1, j_idx - n_fit:j_idx + n_fit + 1])
        idx_j, idx_i = np.meshgrid(np.arange(-n_fit, n_fit + 1), np.arange(-n_fit, n_fit + 1))
        idx_j = idx_j.flatten()
        idx_i = idx_i.flatten()
        p = np.linalg.lstsq(np.array([np.ones(np.shape(idx_i)), idx_i, idx_j, idx_i ** 2, idx_i * idx_j, idx_j ** 2]).T,
                            fit_data.flatten())[0]
        delta_ij = np.array([2 * p[2] * p[3] - p[1] * p[4], 2 * p[5] * p[1] - p[2] * p[4]]) / (
                             p[4] ** 2 - 4 * p[5] * p[3])

    # Sanity check
    if np.any(np.abs(delta_ij) > 1.5):
        delta_ij = np.array([0, 0])

    # Recalculate midpoint and make it relative
    return delta_ij + np.array([i_idx, j_idx]) - mid_points


def sanity_checker(image_paths, blood_test=False):
    """ The main function managing images and tests. """

    result = {'Error': '', 'BcSpot': None, 'MescSpot': None, 'MescProblem': None, 'BloodPresent': blood_test}

    # Image 0
    image_idx = 0
    reference_chambers, result['Error'], _ = find_chambers(image_paths[image_idx])
    if result['Error'] != '':
        return result

    # Image 1
    image_idx = 1
    chambers, result['Error'], result['BloodPresent'] = find_chambers(image_paths[image_idx], blood_test)
    if result['Error'] != '':
        return result
    result['BcSpot'] = detect_spot_bc(chambers[0], reference_chambers[0])
    result['MescProblem'] = detect_badfill_mesc(chambers[1], reference_chambers[1])
    dry_mesc_chamber = chambers[1]

    # Image 2
    image_idx = 2
    chambers, result['Error'], _ = find_chambers(image_paths[image_idx])
    if result['Error'] != '':
        return result
    result['MescSpot'] = detect_spot_mesc(chambers[1], dry_mesc_chamber)
    if result['MescProblem'] == 0:
        result['MescProblem'] = detect_badfill_mesc(chambers[1], reference_chambers[1])

    return result


class Chamber:
    """
    Chamber object to hold location and imagedata for the different chambers.
    """
    Name = ''
    I = np.zeros(0)
    J = np.zeros(0)
    Img = np.zeros((0, 0), np.uint8)
    X = np.zeros((0, 0), np.uint8)
    Y = np.zeros((0, 0), np.uint8)
    R = np.zeros((0, 0), np.uint8)

    def __init__(self, name, image, i_mean, j_mean, i_r, j_r):
        self.Name = name
        self.I = np.round(i_mean + np.arange(-i_r, i_r + 1)).astype('int')
        self.I = self.I[np.logical_and(self.I > -1, self.I < np.size(image, 1))]
        self.J = np.round(j_mean + np.arange(-j_r, j_r + 1)).astype('int')
        self.J = self.J[np.logical_and(self.J > -1, self.J < np.size(image, 1))]
        self.Img = image[self.I[0]:self.I[-1] + 1, self.J[0]:self.J[-1] + 1]
        self.X, self.Y = np.meshgrid(np.linspace(-1, 1, np.size(self.J)), np.linspace(1, -1, np.size(self.I)))
        self.R = np.sqrt(self.X ** 2 + self.Y ** 2)


if __name__ == "__main__":

    # Important Flags
    blood_test = False
    use_local_images = False

    # Load images
    if use_local_images:
        image_folder = '/media/anders/-Anders-3-/Google Drev/BluSense/ImageLibrary_Plasma/Images_Monday_14_08_2017/0'
        image_paths = glob.glob(image_folder + '/*.jpg')
    else:
        parser = argparse.ArgumentParser()
        parser.add_argument('image1')
        parser.add_argument('image2')
        parser.add_argument('image3')
        args = parser.parse_args()
        image_paths = [args.image1, args.image2, args.image3]

    # Check number of images
    n_images = len(image_paths)
    if n_images != 3 and n_images != 5:
        print -1
    if n_images == 5:
        image_paths = [image_paths[0], image_paths[2], image_paths[4]]

    # Run main sanity checks function
    result = sanity_checker(image_paths, blood_test)

    # Check for error
    if not result['Error'] == '':
        print -1

    # Convert result dict. to unique integer. 0 = no problems
    out_bin = [not result['BcSpot'], not result['MescSpot'], result['MescProblem'] > 0,
               result['BloodPresent'] != blood_test]
    out_int = sum([b*2**i for i, b in enumerate(out_bin)])
    print out_int

    # Install instructions for Anaconda 3.6
    # conda update conda
    # conda create -n cv python=2.7.13 anaconda
    # activate cv
    # conda install -n cv -c menpo opencv=2.4.11
    #
    # Other usefull comamnds:
    # source deactivate
    # conda remove -n cv -all
    # conda update -n cv -all
    #
    # Now in PyCharm do
    # Select File, click Settings.
    # In the left pane, enter Project Interpreter in the search box, then click Project Interpreter.
    # In the right pane, click the gear icon, click More...
    # In the Project Interpreters dialog box, click the plus sign +, click Add Local.
    # Use:_/home/anders/anaconda3/envs/cv/bin/pyt
