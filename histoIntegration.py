import sift
import cv2 as cv
import numpy as np
import utils
import segmentation

# Constants
MIN_MATCH_COUNT = 4
MIN_DISTANCE_THRESHOLD = 0.75
RANSAC_REPROJ_THRESHOLD = 5.0

def make_grid(img_puzzle, puzzle_size):
    # Hardcoded puzzle sizes for testing
    puzzle_dimensions = {24:[4, 6], 60:[6,10], 96:[8,12]}
    nrows, ncols = puzzle_dimensions[puzzle_size]

    kernel_size = 30
    bgr_range = 15
    # Get background color
    b, g, r = sift.get_background_color(img_puzzle)
    # Define kernel for morphology closing
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    # Mask and fill pieces to black/white
    puzzle_filled = sift.mask_and_fill(img_puzzle.copy(), b, g, r, kernel, bgr_range)
    # Preform Canny Edge detection on filled pieces
    puzzle_edges = sift.canny_edge_detection(puzzle_filled)
    # Get contours from edges
    contour = sift.get_max_contour(puzzle_edges)

    x, y, puzzwidth, puzzheight = cv.boundingRect(contour)

    gridpixels = []

    # Split puzzle into grid pixel boundaries
    for i in range(0, nrows):
        for j in range(0, ncols):
            # each entry is (xleftedge, xrightedge, ybottomedge, ytopedge)
            utils.save_grid(i*ncols+j, int((((j * puzzwidth / ncols)+x) + ((j * puzzwidth / ncols + puzzwidth / ncols)+x)) / 2), int((((i * puzzheight / nrows)+y) + ((i * puzzheight / nrows + puzzheight / nrows)+y)) / 2), (int)(j * puzzwidth / ncols)+x, (int)(j * puzzwidth / ncols + puzzwidth / ncols)+x, (int)(i * puzzheight / nrows)+y, (int)(i * puzzheight / nrows + puzzheight / nrows)+y)
            gridpixels.append(((int)(j * puzzwidth / ncols)+x, (int)(j * puzzwidth / ncols + puzzwidth / ncols)+x, (int)(i * puzzheight / nrows)+y, (int)(i * puzzheight / nrows + puzzheight / nrows)+y))

    cv.imwrite("Puzzle Examples/GRID.jpg", img_puzzle)
    return gridpixels

def color_demo(img_puzzle, bgr_range, kernel_size, gridlims, gridmatched, piecematched):
    utils.make_dir("Color Matches")
    puzzle_size = utils.dir_size("Pieces")
    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    #storage for histograms
    piece_hist_r = []
    piece_hist_g = []
    piece_hist_b = []
    puzzle_hist_r = []
    puzzle_hist_g = []
    puzzle_hist_b = []

    # Split puzzle into grid and find histogram of each piece
    for i in range(len(gridmatched)):
        if not gridmatched[i]:
            roi = img_puzzle[gridlims[i][2]:gridlims[i][3], gridlims[i][0]:gridlims[i][1]]
            roi_planes = cv.split(roi)
            b_hist = cv.calcHist(roi_planes, [0], None, [256], [0, 255], accumulate=False)
            g_hist = cv.calcHist(roi_planes, [1], None, [256], [0, 255], accumulate=False)
            r_hist = cv.calcHist(roi_planes, [2], None, [256], [0, 255], accumulate=False)
            cv.normalize(b_hist, b_hist, alpha=0, beta=1, norm_type=cv.NORM_MINMAX)
            cv.normalize(g_hist, g_hist, alpha=0, beta=1, norm_type=cv.NORM_MINMAX)
            cv.normalize(r_hist, r_hist, alpha=0, beta=1, norm_type=cv.NORM_MINMAX)
            puzzle_hist_b.append(b_hist)
            puzzle_hist_g.append(g_hist)
            puzzle_hist_r.append(r_hist)
        else:
            puzzle_hist_b.append(0)
            puzzle_hist_g.append(0)
            puzzle_hist_r.append(0)

    # Reading in piece images as a loop
    for i in range(puzzle_size):
        if(not piecematched[i]):
            img_piece_original = cv.imread("Pieces/Piece_{}.png".format(i), cv.IMREAD_COLOR)

            # get background color, add padding
            b,g,r = sift.get_background_color(img_piece_original)
            img_piece = cv.copyMakeBorder(img_piece_original, 50, 50, 50, 50, cv.BORDER_CONSTANT, value=(b, g, r))

            piece_planes = cv.split(img_piece)
            b_hist = cv.calcHist(piece_planes, [0], None, [256], [0, 255], accumulate=False)
            g_hist = cv.calcHist(piece_planes, [1], None, [256], [0, 255], accumulate=False)
            r_hist = cv.calcHist(piece_planes, [2], None, [256], [0, 255], accumulate=False)

            # normalize and append the histogram
            cv.normalize(b_hist, b_hist, alpha=0, beta=1, norm_type=cv.NORM_MINMAX)
            cv.normalize(g_hist, g_hist, alpha=0, beta=1, norm_type=cv.NORM_MINMAX)
            cv.normalize(r_hist, r_hist, alpha=0, beta=1, norm_type=cv.NORM_MINMAX)
            piece_hist_b.append(b_hist)
            piece_hist_g.append(g_hist)
            piece_hist_r.append(r_hist)

            comps = []
            for p in range(puzzle_size):
                if (not gridmatched[p]):
                    fullcomp = cv.compareHist(puzzle_hist_r[p], piece_hist_r[i], cv.HISTCMP_CORREL) + \
                               cv.compareHist(puzzle_hist_g[p], piece_hist_g[i], cv.HISTCMP_CORREL) + \
                               cv.compareHist(puzzle_hist_b[p], piece_hist_b[i], cv.HISTCMP_CORREL)
                    comps.append(fullcomp)
                else:
                    comps.append(-100)
            print("Piece {} and Grid {} are the best match.".format(i, np.argmax(comps)))
            segmentation.spline_matches(img_puzzle, i, np.argmax(comps))
        else:
            piece_hist_b.append(0)
            piece_hist_g.append(0)
            piece_hist_r.append(0)

    return 0
