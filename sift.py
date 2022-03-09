import shutil
import cv2 as cv
import numpy as np
import statistics
import utils
import histoIntegration
# Constants
MIN_MATCH_COUNT = 4
MIN_DISTANCE_THRESHOLD = 0.75
RANSAC_REPROJ_THRESHOLD = 5.0

def sift_demo(img_puzzle, bgr_range, kernel_size):
    utils.make_dir("SIFT Matches")
    puzzle_size = utils.dir_size("Pieces")
    # Assume pieces have been already split from image in grayscale
    imgpath = img_puzzle
    img_puzzle = cv.imread(img_puzzle, cv.IMREAD_COLOR)


    pieces_matched = [False for i in range(puzzle_size)]  # keeps track of what pieces are matched


    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    # Initializing SIFT detector for puzzle
    sift = cv.SIFT_create()
    kp2, des2 = sift.detectAndCompute(img_puzzle, None)

    # Initializing BruteForce matcher
    bf = cv.BFMatcher()

    # array for storing center coordinates with current index of piece
    center_coords = []

    # Reading in images as a loop
    for i in range(puzzle_size):
        img_puzzle_copy = img_puzzle.copy()
        img_piece_original = cv.imread("Pieces/Piece_{}.png".format(i), cv.IMREAD_COLOR)

        # get background color, add padding
        b,g,r = get_background_color(img_piece_original)
        img_piece = cv.copyMakeBorder(img_piece_original, 50, 50, 50, 50, cv.BORDER_CONSTANT, value=(b, g, r))

        # Mask piece based on background color and fill in
        piece_filled = mask_and_fill(img_piece.copy(), b, g, r, kernel, bgr_range)

        # Perform Edge Detection on Solid Image
        piece_edges = canny_edge_detection(piece_filled)

        # Dilate image to connect edges
        piece_edges = cv.dilate(piece_edges, np.ones((3, 3), np.uint8), iterations=1)

        # Get max contour
        piece_contour = get_max_contour(piece_edges)

        # Detecting keypoints of piece & calculating descriptors
        kp1, des1 = sift.detectAndCompute(img_piece, None)

        # Finding k=2 best matches
        matches = bf.knnMatch(des1, des2, k=2)

        # Applying Lowe's ratio test to filter out ambiguous matches
        good_matches = ratio_test(matches, MIN_DISTANCE_THRESHOLD)

        # Check if number of matches found is greater than minimum threshold
        if len(good_matches) >= MIN_MATCH_COUNT:
            # Reshaping keypoints array from both images to preform findHomography
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

            # Find perspective transformation of image with mask specifying inliers/outliers
            M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, RANSAC_REPROJ_THRESHOLD)

            # Using contours from edge detection splines, with max array go [0:] instead of [0]
            pts_spline = np.float32([piece_contour[0:]]).reshape(-1, 1, 2)
            dst_spline = []
            w = h = 0
            # Error check
            if M is not None:
                dst_spline = cv.perspectiveTransform(pts_spline, M)
                # Validate dst spline is not trash by checking area of bounding box
                x,y,w,h = cv.boundingRect(dst_spline)

            if w*h == 0 or w*h > (img_puzzle.shape[0]*img_puzzle.shape[1]*3/puzzle_size):
                print("No accurate match was found - Piece {}".format(i))
                # combo = cv.drawMatches(img_piece, kp1, img_puzzle_copy, kp2, good_matches, None,
                #                        matchColor=(255 - b, 255 - g, 255 - r), singlePointColor=None, matchesMask=None,
                #                        flags=2)

            else:
                # Appending coordinates of circle center where destination spline matched to, with corresponding index
                centerx, centery = get_circ_coords(dst_spline)
                center_coords.append([i, centerx, centery])
                # Retrieve coordinates saved from segmentation step
                original_coords = utils.find_coordinates(i)
                # Draw contour on puzzle image
                img_puzzle_copy = cv.polylines(img_puzzle_copy, [np.int32(dst_spline)], True, (255-b, 255-g, 255-r), 5, cv.LINE_AA)

                # Stack the pieces and puzzle image
                img_contour = cv.imread("Contours/Contour_{}.jpg".format(i), cv.IMREAD_COLOR)
                combo = np.concatenate((img_contour, img_puzzle_copy), axis=0)
                pieces_matched[i] = True


                # Draw arrowed line from piece to contour
                cv.arrowedLine(combo, (int(original_coords[0]),int(original_coords[1])), (centerx, centery+img_puzzle_copy.shape[0]),(255-b, 255-g, 255-r), thickness=5)
                # Write the SIFT match to created directory
                cv.imwrite("SIFT Matches/SIFT_{}.jpg".format(i), combo)

        # If not enough matches
        else:
            print("Not enough keypoints were found - Piece {}: {}/{}".format(i, len(good_matches), MIN_MATCH_COUNT))
            # combo = cv.drawMatches(img_piece, kp1, img_puzzle_copy, kp2, good_matches, None,
            #                        matchColor=(255-b, 255-g, 255-r), singlePointColor=None, matchesMask=None, flags=2)

    # Sort sift matches to correct order
    sort_sift_files(center_coords, puzzle_size)

    # Run Color Histogram on remaining pieces
    numcol, numrow = get_puzzle_dim(center_coords)

    gridlimits = histoIntegration.make_grid(img_puzzle, puzzle_size)

    grid_matched = [False for i in range(puzzle_size)]

    for i, limits in enumerate(gridlimits):
        for coord in center_coords:
            if (coord[1] > limits[0] and coord[1] < limits[1] and coord[2] > limits[2] and coord[2] < limits[3]):
                # this is a match at this grid location
                grid_matched[i] = True


    histoIntegration.color_demo(img_puzzle, bgr_range, kernel_size, gridlimits, grid_matched, pieces_matched)

    # Return number of matches found
    return len(center_coords)

# Returns center coordinate of min enclosing circle of spline
def get_circ_coords(spline):
    (x,y), radius = cv.minEnclosingCircle(spline)
    return int(x), int(y)

# Sorting pieces based on grid of points
def sort_pieces(coords):
    if len(coords) > 0:
        # First need to flip (y,x) to (x,y)
        for i in range(len(coords)):
            tempx = coords[i][1]
            coords[i][1] = coords[i][2]
            coords[i][2] = tempx

        # sort based on X value ascending
        coords.sort(key=lambda x: x[1])

        # Find breakpoints for rows if difference is > average -- More efficient way?
        differences = []
        for i in range(len(coords) - 1):
            dif = coords[i+1][1] - coords[i][1]
            differences.append(dif)

        average_of_differences = sum(differences) / len(differences)

        # making list of rows
        rows = []
        start = 0
        for i in range(len(differences)):
            if (differences[i] > average_of_differences):
                rows.append(coords[start:i+1])
                start = i+1
            elif (i == len(differences)-1):
                rows.append(coords[start:len(coords)])

        numCol = 0
        # For each row, sort based on y value to get left->right
        for i in range(len(rows)):
            rows[i].sort(key=lambda y:y[2])
            numCol = max(numCol, len(rows[i]))

        print("Number rows: ", len(rows))
        print("Number cols:", numCol)
        # Append back together (cute 1 liner)
        sorted_list = [piece for row in rows for piece in row]
        return sorted_list

    else:
        return []

def get_puzzle_dim(coords):
    if len(coords) > 0:
        # First need to flip (y,x) to (x,y)
        for i in range(len(coords)):
            tempx = coords[i][1]
            coords[i][1] = coords[i][2]
            coords[i][2] = tempx

        # sort based on X value ascending
        coords.sort(key=lambda x: x[1])

        # Find breakpoints for rows if difference is > average -- More efficient way?
        differences = []
        for i in range(len(coords) - 1):
            dif = coords[i+1][1] - coords[i][1]
            differences.append(dif)

        average_of_differences = sum(differences) / len(differences)

        # making list of rows
        rows = []
        start = 0
        for i in range(len(differences)):
            if (differences[i] > average_of_differences):
                rows.append(coords[start:i+1])
                start = i+1
            elif (i == len(differences)-1):
                rows.append(coords[start:len(coords)])

        numCol = 0
        # For each row, sort based on y value to get left->right
        for i in range(len(rows)):
            rows[i].sort(key=lambda y:y[2])
            numCol = max(numCol, len(rows[i]))

        return (len(rows), numCol)

    else:
        return (0, 0)

# Get background color by finding most common edge pixel
def get_background_color(img):
    b1 = img[0, :, 0]
    g1 = img[0, :, 1]
    r1 = img[-1, :, 2]
    b2 = img[-1, :, 0]
    g2 = img[-1, :, 1]
    r2 = img[-1, :, 2]
    b3 = img[:, 0, 0]
    g3 = img[:, 0, 1]
    r3 = img[:, 0, 2]
    b4 = img[:, -1, 0]
    g4 = img[:, -1, 1]
    r4 = img[:, -1, 2]
    edge_b = np.concatenate((b1, b2, b3, b4))
    edge_g = np.concatenate((g1, g2, g3, g4))
    edge_r = np.concatenate((r1, r2, r3, r4))
    mode_b = statistics.mode(edge_b)
    mode_g = statistics.mode(edge_g)
    mode_r = statistics.mode(edge_r)

    return int(mode_b), int(mode_g), int(mode_r)

# Fills piece to white with black background
def mask_and_fill(img, b, g, r, kernel, range):

    mask = cv.inRange(img, (b-range, g-range, r-range), (b+range, g+range, r+range))
    cv.bitwise_not(mask, mask)

    #Debugging statements
    # cv.imshow('mask', mask)
    # cv.imshow('filled', cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel))
    # cv.waitKey(0)

    return cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)

# Preform Canny edge detection on solid image
def canny_edge_detection(img):
    return cv.Canny(img, 100, 200)

# Get max contour of edge detected image
def get_max_contour(img):
    # Consider Pixel Threshold Values and Find Contours
    ret, thresh = cv.threshold(img, 200, 255, 0)
    contours, hierarchy = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

    # Take biggest contour only
    return max(contours, key=cv.contourArea)

# Get all contours
def get_contours(img, num):
    # Consider Pixel Threshold Values and Find Contours
    ret, thresh = cv.threshold(img, 200, 255, 0)
    contours, hierarchy = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

    # Sorting by area to exclude small contours accidentally detected
    contours.sort(key=cv.contourArea, reverse=True)
    return contours[:num]

# Preform Lowe's ratio test for the matches
def ratio_test(matches, thresh):
    good_matches = []
    for m, n in matches:
        if m.distance < n.distance * thresh:
            good_matches.append(m)
    return good_matches

# Sorts SIFT Matching image files to be in correct order
def sort_sift_files(coords, num_pieces):
    utils.make_dir("SIFT Sorted")
    order = sort_pieces(coords)

    pieces = [False for i in range(num_pieces)]
    for i in range(len(order)):
        file_num = order[i][0]
        pieces[file_num] = True
        shutil.copy("SIFT Matches/SIFT_{}.jpg".format(file_num), "SIFT Sorted/PieceNum{}.jpg".format(i))

    # Check for pieces that were not in the ordering at put at end
    # unmatched_pieces = len(order)
    # for i in range(num_pieces):
    #     if not pieces[i]:
    #         shutil.copy("SIFT Matches/SIFT_{}.jpg".format(i), "SIFT Sorted/PieceNum{}.jpg".format(unmatched_pieces))
    #         unmatched_pieces += 1
