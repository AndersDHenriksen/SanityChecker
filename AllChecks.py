import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import cv2
from skimage import morphology, restoration, measure, draw, segmentation, filters, transform
import scipy.signal
import scipy.stats.stats
import glob


def find_openings(image_path, blood_also=False):
    """ Take full image and return openings in the label.

	Labels wholes are identified using rough idea of their location,
	the bottom hat algorithm and thresholding.
	In the found openings have an unexpected size an error string will be returned.

    :param image_path: String with path and name of image.
    :param blood_also: Bool, if blood openings should be found.
    :return: List of either 2 or 4 opening objects; String with possible errors.
    """

    # Define future output and settings
    error = ''
    setting = {'BcPositionExpected': (1330, 1823), 'MescPositionExpected': (1347, 2485),
               'Ofc2PositionExpected': (2300, 500), 'BSSPositionExpected': (2160, 2080),
               'CutSide': [400, 400, 400, 800]}
    if blood_also:
        openings = [Opening('BC', setting['BcPositionExpected']), Opening('MESC', setting['MescPositionExpected']),
                    Opening('OFC2', setting['Ofc2PositionExpected']), Opening('BSS', setting['BSSPositionExpected'])]
    else:
        openings = [Opening('BC', setting['BcPositionExpected']), Opening('MESC', setting['MescPositionExpected'])]

    # Define used structural elements i.e. kernels. Note: in matlab rectangle is faster than ellipse
    kernel_r200 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (400, 400))
    kernel_r25 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (50, 50))

    # Load image
    color_image = cv2.imread(image_path)
    image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)

    # Loop through openings
    for idx, opening in enumerate(openings):
        # Initialize constants
        thres = 25
        increase_thres = True
        cut_side = setting['CutSide'][idx]

        # Cut ROI from image and use bottom hat to find darker opening
        cut_range = range(-cut_side, cut_side + 1)
        cut_out = image[[x + opening.PositionExpected[0] for x in cut_range]][:,
                  [x + opening.PositionExpected[1] for x in cut_range]]
        bot_hat_image = cv2.morphologyEx(cut_out, cv2.MORPH_BLACKHAT, kernel_r200)

        # Make mask of middle of cutOut
        opening_mask = np.zeros(np.shape(cut_out), dtype=bool)
        opening_mask[cut_side * 3 / 4: cut_side * 5 / 4,
        cut_side * 3 / 4: cut_side * 5 / 4] = True

        # Try increasing threshold until appropriate sized opening
        while increase_thres:
            thres_image = np.array(bot_hat_image > thres, dtype=np.uint8)
            thres_image = cv2.morphologyEx(thres_image, cv2.MORPH_OPEN, kernel_r25)
            thres_image = cv2.morphologyEx(thres_image, cv2.MORPH_CLOSE, kernel_r25)
            if not np.any(thres_image):
                error = opening.Name + 'opening threshold increased too much. '
                print 'Error: ' + error  # This could be error/exception instead
                continue

            thres_opening = morphology.reconstruction(opening_mask & thres_image, thres_image)
            thres_projection_i = np.argwhere(np.any(thres_opening, axis=0))
            thres_projection_j = np.argwhere(np.any(thres_opening, axis=1))
            span_i = thres_projection_i.item(-1) - thres_projection_i.item(0)
            span_j = thres_projection_j.item(-1) - thres_projection_j.item(0)
            if span_i > cut_side * 5 / 4 | span_j > cut_side * 5 / 4:
                thres *= 1.1
            else:
                increase_thres = False

        # Cut away pixels which are too white
        thres_opening[cut_out > 140] = False

        # Assign image (wiener filtered) and other fields
        imin = np.argwhere(np.any(thres_opening, axis=1)).item(0) + opening.PositionExpected[0] - cut_side
        imax = np.argwhere(np.any(thres_opening, axis=1)).item(-1) + opening.PositionExpected[0] - cut_side
        jmin = np.argwhere(np.any(thres_opening, axis=0)).item(0) + opening.PositionExpected[1] - cut_side
        jmax = np.argwhere(np.any(thres_opening, axis=0)).item(-1) + opening.PositionExpected[1] - cut_side
        # Alternative use wiener filtered image or maybe bilateral filter
        opening.Img = cv2.medianBlur(image[imin:imax, jmin:jmax], 3)
        opening.I = np.arange(imin, imax + 1)
        opening.J = np.arange(jmin, jmax + 1)

        # Check size
        elips_axis = np.array([np.size(opening.I), np.size(opening.J)])
        if any(elips_axis > cut_side + 100) | any(elips_axis < cut_side - 100):
            error = opening.Name + ' opening is wrong size. '
            print 'Error: ' + error

        # Assign rest of opening
        opening.cImg = color_image[imin:imax, jmin:jmax, :]
        opening.X, opening.Y = np.meshgrid(np.linspace(-1, 1, elips_axis[1] - 1), np.linspace(1, -1, elips_axis[0] - 1))

        opening.R = np.sqrt(opening.X ** 2 + opening.Y ** 2)

    return openings, error


def find_same_chamber(opening, ref_chamber, debug=False):
    """ Take opening and reference chamber and return new chamber at same location as reference.

    :param opening: Opening object, from which to extract chamber.
    :param ref_chamber: Chamber object, whose location will be used.
    :param debug: Bool, if True visual plot of new chamber is produced.
    :return: Chamber object with ref_chamber location but image data from opening.
    """

    i_mean = np.mean(ref_chamber.OpnI)
    j_mean = np.mean(ref_chamber.OpnJ)
    i_r = np.size(ref_chamber.OpnI)
    j_r = np.size(ref_chamber.OpnJ)

    if debug:
        plt.figure()
        cv2.ellipse(opening.cImg, (i_mean, j_mean), (i_r, j_r), 0, 0, 360, (255, 0, 0))

    return Chamber(opening, i_mean, j_mean, i_r, j_r)


def find_mesc_chamber(opening, debug=False):
    """ Find/detect MESC chamber in opening object.

	Using the higher intensity at the chamber edges and Hough transform
	the round MESC chamber is detected in opening image data.

    :param opening: Opening object, from which to extract chamber.
    :param debug: Bool, if True visual plot of new chamber is produced.
    :return: Chamber object of MESC chamber.
    """

    # Define settings which chould be changed in the future
    setting = {'UseThinning': True, 'CutOffR': .75}

    # Define kernels used
    kernel_r7 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (14, 14))
    kernel_r141 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (282, 282))

    # Get contrast image
    chamber_contrast = cv2.morphologyEx(opening.Img, cv2.MORPH_TOPHAT, kernel_r141)

    # Get background value
    inner_mask = opening.R < setting['CutOffR']
    background = np.percentile(chamber_contrast[inner_mask], 10)

    # Get contrast mask for hough transform
    contrast_mask = np.logical_and(chamber_contrast > background + 10, inner_mask)

    # Perturb edges to prevent detection here
    radial_mask = np.angle(opening.X + 1j * opening.Y) % (np.pi / 5) < np.pi / 10
    contrast_mask = np.logical_or(contrast_mask, np.logical_and.reduce((
        cv2.dilate(contrast_mask.astype('uint8'), kernel_r7), np.logical_not(inner_mask), radial_mask)))

    if setting['UseThinning']:
        contrast_mask = morphology.thin(contrast_mask)

    # Find circle. Alternatives: skimage.measure.CircleModel & skimage.measure.ransac
    hough_radii = np.arange(145, 151)
    hough_res = transform.hough_circle(contrast_mask, hough_radii)
    _, cx, cy, radii = transform.hough_circle_peaks(hough_res, hough_radii, total_num_peaks=1)
    circle = (cy[0], cx[0], radii[0])

    # Show image output if true debug flag
    if debug:
        plt.figure()
        plt.imshow(opening.cImg)
        ax = plt.gca()
        ellipse = Ellipse(xy=(circle[1::-1]), width=2 * circle[2], height=2 * circle[2], edgecolor='r', fc='None', lw=2)
        ax.add_patch(ellipse)

    # Create chamber object
    return Chamber(opening, circle[0], circle[1], circle[2], circle[2])


def find_bc_chamber(opening, debug=False):
    """ Find/detect BS chamber in opening object.

	Using the edge detection of the chamber edges and two Hough transforms
	the elliptical chamber is detected in opening image data.

    :param opening: Opening object, from which to extract chamber.
    :param debug: Bool, if True visual plot of new chamber is produced.
    :return: Chamber object of BC chamber.
    """

    # Define settings
    setting = {'UseClosing': True, 'UseThinning': False, 'dMove': 40, 'dCirc': 30}

    # Find good radius
    gradient_mask = imgradient(opening.Img) > 30
    outer_radii = np.arange(0.825, 1.025, 0.025)
    count_radii, _ = np.histogram(opening.R[gradient_mask], outer_radii)
    outer_radius = outer_radii[np.argmin(count_radii)]

    # Create mask for hough transform
    edge_mask = np.logical_and.reduce((gradient_mask, opening.R > .5, opening.R < outer_radius))

    # Adjust mask
    if setting['UseClosing']:
        kernel_r3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (6, 6))
        edge_mask = cv2.morphologyEx(edge_mask.astype('uint8'), cv2.MORPH_CLOSE, kernel_r3)
    if setting['UseThinning']:
        edge_mask = morphology.thin(edge_mask)

    # The following procedure will try to find two cirles, from the bottom/top part of the edge_mask respectively.
    # As the BC chamber is not round on the images, the BC chamber will be the ellipse like overlap of the two circles.
    # If doubts then the bottom circle is trusted more.

    # Define OK range the two circles
    mid_point_top = np.tile(np.array(np.shape(edge_mask)) / 2 + np.array([setting['dCirc'], 0]), [2, 1])
    mid_point_bottom = np.tile(np.array(np.shape(edge_mask)) / 2 - np.array([setting['dCirc'], 0]), [2, 1])
    d_move = np.tile(np.array([-setting['dMove'], setting['dMove']]), [2, 1]).T
    good_range = [mid_point_top + d_move, mid_point_bottom + d_move]

    # Split edge_mask to top/lower part
    top_edge = np.zeros(np.shape(edge_mask), dtype=bool)
    bottom_edge = np.zeros(np.shape(edge_mask), dtype=bool)
    split_idx = int(round(np.size(edge_mask, 1) / 2))
    top_edge[0:split_idx, 200:] = edge_mask[0:split_idx, 200:]
    bottom_edge[split_idx:, 200:] = edge_mask[split_idx:, 200:]

    # Do Hough detection at upper/lower part & combine
    set_top, circle_top = half_hough_detection(top_edge, good_range[0])
    set_bottom, circle_bottom = half_hough_detection(bottom_edge, good_range[1])
    set_combined = np.logical_and(set_top, set_bottom)

    # Get ellipse cemtroid and axis
    props = measure.regionprops(set_combined.astype('uint8'))
    i_mean = props[0]['centroid'][0]
    j_mean = props[0]['centroid'][1]
    i_r = props[0]['minor_axis_length'] / 2
    j_r = props[0]['major_axis_length'] / 2

    # Trust lower circle if outside normal range
    if i_r < 150 or i_r > 174:
        i_mean -= (165 - i_r)
        i_r = 165
    if j_r < 160 or j_r > 200:
        j_r = 183

    if debug:
        plt.figure()
        ax = plt.subplot(1, 3, 1)
        plt.imshow(top_edge)
        ellipse = Ellipse(xy=(circle_top[1::-1]), width=2 * circle_top[2], height=2 * circle_top[2], edgecolor='r',
                          fc='None', lw=2)
        ax.add_patch(ellipse)
        ax = plt.subplot(1, 3, 2)
        plt.imshow(bottom_edge)
        ellipse = Ellipse(xy=(circle_bottom[1::-1]), width=2 * circle_bottom[2], height=2 * circle_bottom[2],
                          edgecolor='r', fc='None', lw=2)
        ax.add_patch(ellipse)
        ax = plt.subplot(1, 3, 3)
        plt.imshow(opening.cImg)
        ellipse = Ellipse(xy=(j_mean, i_mean), width=2 * j_r, height=2 * i_r, edgecolor='r', fc='None', lw=2)
        ax.add_patch(ellipse)

    return Chamber(opening, i_mean, j_mean, i_r, j_r)


def imgradient(img):
    """ Calculates the (Sobel) gradient magnitude of the image."""
    return np.sqrt(cv2.Sobel(img, cv2.CV_64F, 1, 0) ** 2 + cv2.Sobel(img, cv2.CV_64F, 0, 1) ** 2)


def half_hough_detection(mask, good_range):
    """ Detect circle using half the image.

	Using Hough transform the best circle corresponding to edge mask is found.
	If this circle is outside the acceptable range. Then fix j_center and radius and
	loop through i_center to find highest overlap.

    :param mask: edge mask where to find the circle
    :param good_range: acceptable range for circle center coordinates
    :return: boolean np-array where True means in cirlce; tuple with circle's (i,j)_center and r
    """

    radius = 190

    # Find circle in mask
    hough_radii = np.linspace(190, 195, 2)
    hough_res = transform.hough_circle(mask, hough_radii)
    _, cx, cy, radii = transform.hough_circle_peaks(hough_res, hough_radii, total_num_peaks=1)
    circle = (cy[0], cx[0], radii[0])

    # If circle outside good_range, sweep circle along i and look for overlap with mask
    if circle[0] < good_range[0, 0] | circle[0] > good_range[1, 0] | \
            circle[1] < good_range[0, 1] | circle[1] > good_range[1, 1]:
        i_range = np.arange(good_range[0, 0], good_range[1, 0] + 1)
        i_mid = int(round(np.mean(i_range)))
        j_mid = int(round(np.size(mask, 1) / 2))
        overlap = np.zeros(np.shape(i_range))
        r_cir, c_cir = draw.circle_perimeter(i_mid, j_mid, radius)
        for i in range(np.size(i_range)):
            r_cor = r_cir + i_range[i] - i_mid
            c_cor = c_cir[np.logical_and(r_cor > -1, r_cor < np.size(mask, 1))]
            r_cor = r_cor[np.logical_and(r_cor > -1, r_cor < np.size(mask, 1))]
            overlap[i] = np.sum(mask[r_cor, c_cor])

        if np.max(overlap) > 20:
            circle = (j_mid, i_range[np.argmax(overlap)], radius)  # TODO check j_mid first
        else:
            circle = (j_mid, i_mid, radius)

    # Make bw set image from circle
    r_cir, c_cir = draw.circle(circle[0], circle[1], circle[2], shape=mask.shape)
    set_cir = np.zeros(mask.shape, dtype=bool)
    set_cir[r_cir, c_cir] = True
    return set_cir, circle


def bc_spot_detection(chamber, debug=False):
    """ Detect reagent spot in BS chamber.

	Extract intensity histogram for the part of the BC chamber in the shadow of the label.
	Calculate features based on this histogram, and use a linear classifier to determine
	if reagent spot is present.

    :param chamber: BS chamber to detect spot in.
    :param debug: Bool, if True visual plot to understand algorithm steps.
    :return: Bool, if True reagent spot was detected.
    """

    setting = {'UseRed': True, 'UseLR': False}
    roi_poly = np.array([(68, 43), (96, 156), (177, 234), (273, 265), (214, 302), (132, 294), (16, 240),
                         (10, 120), (68, 43)], dtype=np.int32)  # TODO are two [[ and ]] needed
    conclusions = ['Spot NOT detected in BC chamber.', 'Spot detected in BC chamber.']

    # Define mask for extraction the points inside the ROI
    in_shadow_mask = np.zeros(chamber.Img.shape, dtype=bool)
    cv2.fillConvexPoly(in_shadow_mask, roi_poly, True)

    if setting['UseRed']:
        usefull_image = chamber.cImg[:, :, 3][in_shadow_mask]  # TODO check channel 3 is red
    else:
        usefull_image = chamber.Img[in_shadow_mask]

    # Calculate histogram, feature and binary activations of ROI
    useful_count = np.histogram(usefull_image, np.arange(np.min(usefull_image), np.max(usefull_image) + 1),
                                density=True)
    features = image_roi_features(usefull_image)
    activations = np.array([features[1] > 0.087, features[2] < 12, features[3] < 15, features[4] < 16,
                            features[6] < 5.4, features[7] > 1.55, features[8] > 10, features[9] < 0.006])

    # Check if spot based on activation.
    if setting['UseLR']:
        weights = np.array([236.60, -126.46, -77.874, -97.041, 130.19, -66.666, -135.72, -65.798, -57.961])
        has_spot = np.insert(activations, 0, 1).dot(weights) > 0
    else:
        weights = np.array([.7, .7, .7, .4, .4, 1, 1, .4])
        has_spot = activations.dot(weights) <= 2

    # Check if spot based on histogram shape, i.e. look for two separate peaks
    if not has_spot and np.size(useful_count) > 8:
        i_range = range(4, np.size(useful_count) - 3)
        for i in i_range:
            idx_1 = np.argmax(useful_count[:i - 3])
            peak_1 = useful_count[idx_1]
            idx_2 = np.argmax(useful_count[i + 4:])
            peak_2 = useful_count[i + 4:][idx_2]
            has_spot = has_spot | ((min([peak_1, peak_2]) > useful_count[i] & peak_2 / peak_1 > .1) &
                                   (peak_1 / peak_2 > .1) & (
                                       min([peak_1, peak_2]) / min(useful_count[peak_1:peak_2 + i + 3]) > 2))

    print conclusions[has_spot]
    if debug:
        plt.figure()
        plt.subplot(1, 2, 1)
        plt.imshow(chamber.cImg)
        plt.Polygon(roi_poly,
                    facecolor=None)  # https://stackoverflow.com/questions/44025403/how-to-use-matplotlib-path-to-draw-polygon
        plt.subplot(1, 2, 2)
        plt.plot(np.arange(np.min(usefull_image), np.max(usefull_image)), useful_count)
        plt.title(conclusions[has_spot])

    return has_spot


def image_roi_features(image_vector):
    """ Calculate features describing the histogram of the image_vector input.

    :param image_vector: array/list of intensitiy values inside the ROI.
    :return: list with 10 feature values.
    """

    # Calculate histogram and peak value
    image_hist = np.histogram(image_vector, np.arange(0, 256), density=True)
    peak = max(image_hist)

    # Calculate features
    feature = [] * 10
    feature[0] = np.median(image_vector)
    feature[1] = peak
    feature[2] = max_span(image_hist > .4 * peak)
    feature[3] = max_span(image_hist > .3 * peak)
    feature[4] = max_span(image_hist > .2 * peak)
    feature[5] = np.mean(image_vector)
    feature[6] = np.std(image_vector)
    feature[7] = scipy.stats.stats.skew(image_vector)
    feature[8] = scipy.stats.stats.kurtosis(image_vector)

    # Calculate distance to similar gauss
    gauss = 1 / np.sqrt(2 * np.pi * feature[6] ** 2) * np.exp(
        -(np.arange(0, 256) - feature[5] ** 2 / (2 * feature[6] ** 2)))
    feature[9] = np.sum((image_hist - gauss) ** 2)  # TODO check this
    return feature


def max_span(a):
    """ Return the lenght/span of true elements in input. """
    return np.argwhere(a)[-1] - np.argwhere(a)[0]


def bss_blood_detection(opening, debug=False):
    """ Detect if blood is present in BSS chamber.

	Detects blood if a harcoded Region-Of-Interest (ROI) is relatively darker than opening.

    :param opening: BSS opening to detect blood in.
    :param debug: Bool, if True visual plot to understand algorithm steps.
    :return: Bool, True if blood was detected.
    """

    setting = {'Ratio': 0.25}
    roi_poly = np.array([(464, 485), (498, 585), (576, 507), (626, 247), (579, 155), (503, 235), (464, 485)],
                        dtype=np.int32)
    conclusions = ['Blood NOT detected in BSS chamber.', 'Blood detected in BSS chamber.']

    # Use polygon mask to see if region is darker
    mask = np.zeros(opening.Img.shape, dtype=bool)
    cv2.fillConvexPoly(mask, roi_poly, True)
    blood_present = np.mean(opening.Img[mask]) / np.mean(opening.Img) < setting['Ratio']

    print conclusions[blood_present]
    if debug:
        plt.figure()
        plt.imshow(opening.Img)
        plt.Polygon(roi_poly,
                    facecolor=None)  # https://stackoverflow.com/questions/44025403/how-to-use-matplotlib-path-to-draw-polygon
        plt.title(conclusions[blood_present])
    return blood_present


def ofc2_blood_detection(opening, debug=False):
    """ Detect if blood is present in OFC2 chamber.

	Detects blood if a harcoded Region-Of-Interest (ROI) is relatively darker than opening.

    :param opening: OFC2 opening to detect blood in.
    :param debug: Bool, if True visual plot to understand algorithm steps.
    :return: Bool, True if blood was detected.
    """

    setting = {'Ratio': 0.5}
    roi_poly = np.array([(237, 326), (322, 363), (349, 309), (333, 84), (237, 326)], dtype=np.int32)
    conclusions = ['Blood NOT detected in OFC2 chamber.', 'Blood detected in OFC2 chamber.']

    # Use polygon mask to see if region is darker
    mask = np.zeros(opening.Img.shape, dtype=bool)
    cv2.fillConvexPoly(mask, roi_poly, True)
    blood_present = np.mean(opening.Img[mask]) / np.mean(opening.Img) < setting['Ratio']

    print conclusions[blood_present]
    if debug:
        plt.figure()
        plt.imshow(opening.Img)
        plt.Polygon(roi_poly,
                    facecolor=None)  # https://stackoverflow.com/questions/44025403/how-to-use-matplotlib-path-to-draw-polygon
        plt.title(conclusions[blood_present])
    return blood_present


def mesc_spot_detection(chamber, debug=False):
    """ Detect beads spot in MESC chamber.

	Look for a magnetic beads spot either using a computer vision based intensitiy method,
	or using histogram of the middle chamber intensities and linear classifier.

    :param chamber: MESC chamber to detect spot in.
    :param debug: Bool, if True visual plot to understand algorithm steps.
    :return: Bool, if True beads spot was detected.
    """

    setting = {'UseLinearCorrection': True, 'UseLR': False}
    conclusions = ['Beads NOT detected in MESC chamber.', 'Beads detected in MESC chamber.']

    # Define contrast image and inner mask
    kernel_r110 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (220, 220))
    chamber_contrast = cv2.morphologyEx(chamber.Img, cv2.MORPH_TOPHAT, kernel_r110)
    inner_mask = chamber.R < 0.8

    # Remove any linear slope in contrast mask, if contrast mask is too bright in SW corner
    if setting['UseLinearCorrection']:
        coef = np.linalg.lstsq(np.array([np.ones(chamber.X[inner_mask].shape, dtype=float),
                                         chamber.X[inner_mask].astype('f'), chamber.Y[inner_mask].astype('f')]),
                               chamber_contrast[inner_mask])
        chamber_contrast = chamber_contrast - min([0, coef[1]]) * chamber.X - min(
            [0, coef[2]]) * chamber.Y  # TODO check this can be handled as uint8

    # Use Logistic regression and histogram to determine if beads spot
    if setting['UseLR']:
        usefull_contrast = chamber_contrast[chamber.R < 0.68]
        features = image_roi_features(usefull_contrast)
        weights = np.array([-25.193, -2.131, 6.5302, 0.17148, 0.20268, 1.5053, 4.0306, -1.4134, -2.8493, 0.13126,
                            257.1])
        has_beads = np.insert(features, 0, 1).dot(weights) > 0

        useful_count = np.histogram(usefull_contrast, np.arange(np.min(usefull_contrast), np.max(usefull_contrast) + 1),
                                    density=True)

        # Look for double peak in histogram
        if not has_beads and np.size(useful_count) > 8:
            i_range = range(4, np.size(useful_count) - 3)
            for i in i_range:
                idx_1 = np.argmax(useful_count[:i - 3])
                peak_1 = useful_count[idx_1]
                idx_2 = np.argmax(useful_count[i + 4:])
                peak_2 = useful_count[i + 4:][idx_2]
                has_beads = has_beads | ((min([peak_1, peak_2]) > useful_count[i] & peak_2 / peak_1 > .1) &
                                         (peak_1 / peak_2 > .1) & (
                                             min([peak_1, peak_2]) / min(useful_count[peak_1:peak_2 + i + 3]) > 2))

        print conclusions[has_beads]
        if debug:
            plt.figure()
            plt.subplot(1, 2, 1)
            plt.imshow(chamber_contrast)
            plt.subplot(1, 2, 2)
            plt.plot(np.arange(np.min(usefull_contrast), np.max(usefull_contrast)), useful_count)
            plt.title(conclusions[has_beads])
        return has_beads

    # Get background value of chamber_contrast
    inner_contrast = chamber_contrast[inner_mask]
    background = np.mean(inner_contrast < np.percentile(inner_contrast, 20))

    # Loop through different contrasts-threshold to find most likely bead pattern
    center_mask = chamber.R < 0.2
    spot_sizes = [] * 8
    contrast_masks = [] * 8
    thresholds = range(3, 11)
    for i in range(0, len(thresholds)):
        contrast_masks[i] = np.logical_or(chamber_contrast > background + thresholds[i], np.invert(inner_mask))
        segmentation.clear_border(contrast_masks[i], in_place=True)
        contrast_masks[i] = morphology.reconstruction(center_mask & contrast_masks[i], contrast_masks[i])
        spot_sizes[i] = np.sum(contrast_masks[i])

    # Assign contrast_mask to largest spotsize or large contrast region
    best_thres = np.argmax(spot_sizes)
    contrast_mask = np.logical_or(contrast_masks[best_thres],
                                  np.logical_and(chamber_contrast > background + 10, inner_mask))
    contrast_mask = morphology.reconstruction(center_mask & contrast_mask, contrast_mask)

    # Beads are present if largest object is big enough
    contrast_contours = cv2.findContours(contrast_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    spot_size = max(cv2.contourArea(contrast_contours))
    has_beads = spot_size > (4800 + 1000 * (thresholds[best_thres] < 5))

    # Do erosion, if area has not changed much investigate further
    kernel_r3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (6, 6))
    eroded_mask = cv2.erode(contrast_masks, kernel_r3)
    eroded_contours = cv2.findContours(eroded_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    eroded_size = max(cv2.contourArea(eroded_contours))
    solidity = eroded_size / spot_size

    # Investigate by extending the mask while keeping lower bound on contrast or intensity
    if not has_beads & ((spot_size > 2500 & solidity > .5) | (spot_size > 4500 & solidity > .45)):
        ext_contrast = chamber_contrast >= np.min(chamber_contrast[contrast_mask])
        ext_contrast = morphology.reconstruction(contrast_mask & ext_contrast, ext_contrast)
        ext_intensity = chamber.Img >= np.min(chamber.Img[contrast_mask])
        ext_intensity = morphology.reconstruction(contrast_mask & ext_intensity, ext_intensity)
        ext_mask = np.logical_and(inner_mask, np.logical_or(ext_contrast, ext_intensity))
        spot_size = np.sum(ext_mask)
        has_beads = spot_size > (4800 + 1000 * (thresholds[best_thres] < 5))

    print conclusions[has_beads]
    if debug:
        plt.figure()
        plt.subplot(1, 2, 1)
        plt.imshow(chamber.Img)
        plt.subplot(1, 2, 2)
        plt.imshow(contrast_mask)
        plt.title(conclusions[has_beads])
    return has_beads


def mesc_overflow_detection(chamber, reference_chamber, debug=False):
    """ Detect MESC problem, either bubble or overflow.

	The image data from chamber and reference chamber is aligned and subtracted to make a diff image.
	In this diff image veritcal lines of high intensity (overflow)
	and bubbles whose edge are intensity than the background are both detected.

    :param chamber: MESC chamber object from image 2 or 3.
    :param reference_chamber: MESC chamber object before liquid, i.e. from image 1.
    :param debug: Bool, if True visual plot to understand algorithm steps.
    :return: int error code. 0 = no problems. 1 = premature transfer. 2 = bubble.
    """

    setting = {'RCutOff': .95}
    conclusions = ['MESC stayed dry.', 'MESC overflow.', 'MESC bubble.']

    # Calculate translation between images and overlaps
    chm_img, ref_img = get_overlap_images(chamber.Img, reference_chamber.Img)
    translation = corr2d(chm_img, ref_img)
    # TODO Translation = -[Translation(2), Translation(1)] might be needed
    r = get_overlap_images(chamber.R, reference_chamber.Img)
    x = get_overlap_images(chamber.X, reference_chamber.Img)
    r = get_overlap_images(r, ref_img, translation)
    x = get_overlap_images(x, ref_img, translation)
    chm_img, ref_img = get_overlap_images(chm_img, ref_img, translation)

    # Calculate difference between images, and row averages
    diff_img = chm_img - ref_img
    diff_img[r > setting['RCutOff']] = 0
    average_diff = np.sum(diff_img, axis=1) / np.sum(r > setting['RCutOff'], axis=1)

    # Overflow is high intensity line and not last pixels
    max_idx = np.argmax(average_diff)
    mesc_overflow = average_diff[max_idx] > 30 & np.sum(average_diff[max_idx:] > 0) > 4

    # Look for mesc bubble at certain position and size
    if not mesc_overflow & np.mean(diff_img[r < .9]) < 9:
        bubble_thres = min(15, 2 * np.mean(diff_img) + 5)
        bubble_mask = np.logical_and.reduce((diff_img < bubble_thres, r < .9, x < .25))
        bubble_contours = cv2.findContours(bubble_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        bubble_areas = cv2.contourArea(bubble_contours)
        bubble_area = np.sum(bubble_areas[np.logical_and(bubble_areas > 400, bubble_areas < 20000)])
        mesc_overflow = 2 * (bubble_area > 250)

    print conclusions[mesc_overflow]
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
    translation = np.round(translation)

    # Define outputs to prevent changing inputs
    out_img1 = img1.copy()
    out_img2 = img2.copy()

    # Find index overlap for each dimension
    for d in range(2):
        size1 = np.size(img1, d)
        size2 = np.size(img2, d)
        size_move = round((size2 - size1) / 2)
        side1_idx = np.array(set(range(size1)) & set(np.arange(0, size2) - size_move + translation[d]))
        side2_idx = np.array(set(np.arange(0, size1) + size_move - translation[d]) & set(range(size2)))
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
    xcorr2d = scipy.signal.fftconvolve(img1, img2[::-1, ::-1])
    # TODO check if similar to XCORR2D if not: https://github.com/keflavich/agpy/blob/master/agpy/cross_correlation.py

    # Calculate image midpoints
    mid_points = np.floor(np.array(np.shape(img1)) / 2)

    # Crop out 25x25 pixels around midpoint of xcorr result
    xcorr2d_crop = xcorr2d[mid_points[0] - max_movement:mid_points[0] + max_movement + 1,
                   mid_points[1] - max_movement:mid_points[1] + max_movement + 1]

    # Get maximum indexes and recalculate to full image coordinates
    i_idx, j_idx = np.unravel_index(xcorr2d_crop.argmax(), xcorr2d_crop.shape)
    i_idx = i_idx + mid_points[0] - max_movement
    j_idx = j_idx + mid_points[1] - max_movement  # TODO is -1 needed?

    n_fit = 3
    delta_ij = np.array([0, 0])
    if i_idx >= n_fit & j_idx >= n_fit & i_idx + n_fit < np.size(xcorr2d, 1) & j_idx + n_fit < np.size(xcorr2d, 2):
        # Fit data to p(0) + p(1)*x + p(2)*y + p(3)*x^2 + p(4)*xy + p(5)*y^2 and get top point
        fit_data = np.log(xcorr2d[i_idx - n_fit:i_idx + n_fit, j_idx - n_fit:j_idx + n_fit])
        idx_i, idx_j = np.meshgrid(np.arange(-n_fit, n_fit + 1), np.arange(-n_fit, n_fit + 1))  # TODO check this
        p = np.linalg.lstsq(np.array([np.ones(idx_i), idx_i, idx_j, idx_i ** 2, idx_i * idx_j, idx_j ** 2]), fit_data)
        delta_ij = np.array([2 * p[5] * p[1] - p[2] * p[4], 2 * p[2] * p[3] - p[1] * p[4]]) / (
        p[4] ** 2 - 4 * p[5] * p[3])

    # Sanity check
    if np.any(np.abs(delta_ij) > 1.5):  # TODO why is this marked incorrect
        delta_ij = np.array([0, 0])

    # Recalculate midpoint and make it relative
    return delta_ij + np.array([i_idx, j_idx]) - mid_points


def main(blood_test=False):
    """ The main function managing images and tests. """

    result = dict()
    result['Error'] = ''

    # Windows: image_folder = 'E:\Google Drev\BluSense\Image Library_PlasmaSerum\Correct procedure_1'
    # Windows2: image_folder = 'C:\Users\310229518\Google Drive\BluSense\Image Library_PlasmaSerum\Correct procedure_1'
    # Linux: image_folder = '/media/anders/-Anders-3-/Google Drev/BluSense/Image Library_PlasmaSerum/Correct procedure_1'
    image_folder = '/media/anders/-Anders-3-/Google Drev/BluSense/Image Library_PlasmaSerum/Correct procedure_1'
    image_paths = glob.glob(image_folder + '/*.jpg')
    n_images = len(image_paths)
    if (n_images != 3) & (n_images != 5):
        result['Error'] = 'Not 3 or 5 images. '
        print 'Error: ' + result['Error']
        # return Result

    # Image 0
    image_idx = 0
    openings, error = find_openings(image_paths[image_idx], blood_test)
    reference_opening_shape = np.array(openings[1].Img.shape)
    result['Error'] = result['Error'] + error
    bc_chamber = find_bc_chamber(openings[0])
    reference_mesc_chamber = find_mesc_chamber(openings[1])
    result['BcSpot'] = bc_spot_detection(bc_chamber)
    result['MescSpot'] = mesc_spot_detection(reference_mesc_chamber)

    # Image 1
    image_idx = 1
    openings, error = find_openings(image_paths[image_idx], blood_test)
    result['Error'] = result['Error'] + error
    if np.any(np.abs(np.array(openings[1].Img.shape) - reference_opening_shape) > 20):  # TODO why is this marked
        mesc_chamber = find_mesc_chamber(openings[1])
    else:
        mesc_chamber = find_same_chamber(openings[1], reference_mesc_chamber)
    result['MescProblem'] = mesc_overflow_detection(mesc_chamber, reference_mesc_chamber)
    if blood_test:
        result['BloodPresent'] = ofc2_blood_detection(openings[2]) & bss_blood_detection(openings[3])

    print 'Sanity checker finished'
    return result.items()


class Opening:
    Name = ''
    PositionExpected = (0, 0)
    I = np.zeros(0)
    J = np.zeros(0)
    Img = np.zeros((0, 0), np.uint8)
    cImg = np.zeros((0, 0, 3), np.uint8)
    X = np.zeros((0, 0), np.uint8)
    Y = np.zeros((0, 0), np.uint8)
    R = np.zeros((0, 0), np.uint8)

    def __init__(self, name, expected_position):
        self.Name = name
        self.PositionExpected = expected_position


class Chamber:
    Name = ''
    OpnI = np.zeros(0)
    OpnJ = np.zeros(0)
    Img = np.zeros((0, 0), np.uint8)
    cImg = np.zeros((0, 0, 3), np.uint8)
    X = np.zeros((0, 0), np.uint8)
    Y = np.zeros((0, 0), np.uint8)
    R = np.zeros((0, 0), np.uint8)

    def __init__(self, opening, i_mean_opn, j_mean_opn, i_r, j_r):
        self.Name = opening.Name
        self.OpnI = np.round(i_mean_opn + np.arange(-i_r, i_r + 1)).astype('int')
        self.OpnI = self.OpnI[np.logical_and(self.OpnI > -1, self.OpnI < np.size(opening.Img, 1))]
        self.OpnJ = np.round(j_mean_opn + np.arange(-j_r, j_r + 1)).astype('int')
        self.OpnJ = self.OpnJ[np.logical_and(self.OpnJ > -1, self.OpnJ < np.size(opening.Img, 1))]
        self.Img = opening.Img[self.OpnI[0]:self.OpnI[-1], self.OpnJ[0]:self.OpnJ[-1]]
        if np.size(opening.cImg) > 0:
            self.cImg = opening.cImg[self.OpnI[0]:self.OpnI[-1], self.OpnJ[0]:self.OpnJ[-1]]
        self.X, self.Y = np.meshgrid(np.linspace(-1, 1, np.size(self.OpnI)), np.linspace(1, -1, np.size(self.OpnJ)))
        self.R = np.sqrt(self.X ** 2 + self.Y ** 2)


if __name__ == "__main__":
    main()

    # Install instructions for Anaconda 3.6
    # conda update conda
    # conda create -n cv python=2.7.13 anaconda
    # activate cv
    # conda install -n cv -c menpo opencv=2.4.11
    #
    # Other usefull comamnds:
    # source deactivate
    # conda remove -n cv -all
    #
    # Now in PyCharm do
    # Select File, click Settings.
    # In the left pane, enter Project Interpreter in the search box, then click Project Interpreter.
    # In the right pane, click the gear icon, click More...
    # In the Project Interpreters dialog box, click the plus sign +, click Add Local.
    # Use:_/home/anders/anaconda3/envs/cv/bin/pyt