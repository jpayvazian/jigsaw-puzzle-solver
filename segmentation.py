import cv2 as cv
import numpy as np
import sift
import utils

def segment_pieces(img, puzzle_size, bgr_range, kernel_size):
    utils.make_dir("Pieces")
    utils.make_dir("Contours")

    # Read image *IN COLOR*
    img_pieces = cv.imread(img, cv.IMREAD_COLOR)

    # Get background color
    b,g,r = sift.get_background_color(img_pieces)
    # Define kernel for morphology closing
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    # Mask and fill pieces to black/white
    pieces_filled = sift.mask_and_fill(img_pieces.copy(), b, g, r, kernel, bgr_range)
    # Preform Canny Edge detection on filled pieces
    pieces_edges = sift.canny_edge_detection(pieces_filled)
    # Dilate image to connect edges
    pieces_edges = cv.dilate(pieces_edges, np.ones((3, 3), np.uint8), iterations=1)
    # Get contours from edges
    contours = sift.get_contours(pieces_edges, puzzle_size)

    # Iterate through each contour in the image to segment it
    image_number = 0
    for c in contours:
        # Highlight the contour and put in directory for use later
        highlight_contour(c, image_number, b, g, r, img_pieces.copy())
        centerx, centery = sift.get_circ_coords(c)
        utils.save_coordinates(image_number, centerx, centery)
        # Convex hull for tighter segmentation around corners
        hull = cv.convexHull(c)

        # Crop the bounding rect
        rect = cv.boundingRect(hull)
        cropped = img_pieces[rect[1]: rect[1] + rect[3], rect[0]: rect[0] + rect[2]].copy()
        # make mask
        hull = hull - hull.min(axis=0)
        mask = np.zeros(cropped.shape[:2], np.uint8)
        cv.drawContours(mask, [hull], -1, (255, 255, 255), -1)
        # do bit-op
        dst = cv.bitwise_and(cropped, cropped, mask=mask)
        # add the background color
        bgr = np.full((cropped.shape[0], cropped.shape[1], 3), (b, g, r), np.uint8)
        color_crop = cv.bitwise_or(bgr, bgr, mask=cv.bitwise_not(mask))
        dst2 = color_crop + dst

        # Write the segmented piece to new file in created directory
        cv.imwrite("Pieces/Piece_{}.png".format(image_number), dst2)
        image_number += 1

# Runs just the mask and fill operation for updating the UI
def adjust_mask(img_path, bgr_range, kernel_size):
    # New directory for UI to grab image
    utils.make_dir("Puzzle Mask")

    img = cv.imread(img_path, cv.IMREAD_COLOR)
    # Getting background color
    b, g, r = sift.get_background_color(img)
    # Define kernel for morphology closing
    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    # Run mask and fill and write to directory
    mask = sift.mask_and_fill(img.copy(), b, g, r, kernel, bgr_range)
    cv.imwrite("Puzzle Mask/Mask.jpg", mask)

# Outlines one piece contour in the image of pieces scattered
def highlight_contour(contour, num, b, g, r, img):
    cv.drawContours(img, contour, -1, (255-b,255-g,255-r), 5)
    cv.imwrite("Contours/Contour_{}.jpg".format(num), img)

# Resizes two images to be the same size by adding border paddings
def resize_images(img1, img2):
    img_pieces = cv.imread(img1, cv.IMREAD_COLOR)
    img_puzzle = cv.imread(img2, cv.IMREAD_COLOR)

    b1, g1, r1 = sift.get_background_color(img_pieces)
    b2, g2, r2 = sift.get_background_color(img_puzzle)

    dif_height = img_pieces.shape[0] - img_puzzle.shape[0]
    dif_length = img_pieces.shape[1] - img_puzzle.shape[1]

    # If difference is odd, +- 1 to difference
    if dif_length % 2 == 0:
        padding_l = padding_r = int(dif_length / 2)
    else:
        padding_l, padding_r = int((dif_length + 1) / 2), int((dif_length - 1) / 2)
    if dif_height % 2 == 0:
        padding_t = padding_b = int(dif_height / 2)
    else:
        padding_t, padding_b = int((dif_height + 1) / 2), int((dif_height - 1) / 2)

    # Checking which image needs to be resized
    if padding_l and padding_r >= 0:
        img_puzzle = cv.copyMakeBorder(img_puzzle, 0, 0, padding_l, padding_r, cv.BORDER_CONSTANT, value=(b2,g2,r2))
    else:
        img_pieces = cv.copyMakeBorder(img_pieces, 0, 0, abs(padding_l), abs(padding_r), cv.BORDER_CONSTANT, value=(b1,g1,r1))

    if padding_t and padding_b >= 0:
        img_puzzle = cv.copyMakeBorder(img_puzzle, padding_t, padding_b, 0, 0, cv.BORDER_CONSTANT, value=(b2,g2,r2))
    else:
        img_pieces = cv.copyMakeBorder(img_pieces, abs(padding_t), abs(padding_b), 0, 0, cv.BORDER_CONSTANT, value=(b1,g1,r1))

    cv.imwrite(img1, img_pieces)
    cv.imwrite(img2, img_puzzle)

# Draws arrows from piece to its matches
def spline_matches(img1, num, grid):
    utils.make_dir("Spline Matches")
    img_pieces = cv.imread("Contours/Contour_{}.jpg".format(num), cv.IMREAD_COLOR)
    img_puzzle = img1 #cv.imread(img1, cv.IMREAD_COLOR)
    combo = np.concatenate((img_pieces, img_puzzle), axis=0)

    origin = utils.find_coordinates(num)
    b,g,r = sift.get_background_color(img_pieces)

    # Draw arrows to all potential matches
    # for i in range(len(matches)):
    coords = utils.find_grid(grid)

    cv.arrowedLine(combo,(int(origin[0]), int(origin[1])), (int(coords[0]),int(coords[1])+img_puzzle.shape[0]),(255-b, 255-g, 255-r), thickness=5)
    cv.rectangle(combo, (int(coords[2]), int(coords[4])+img_puzzle.shape[0]), (int(coords[3]), int(coords[5])+img_puzzle.shape[0]), (255-b, 255-g, 255-r), thickness=5)

    s = utils.dir_size("SIFT Sorted")
    cv.imwrite("SIFT Sorted/PieceNum{}.jpg".format(s),combo)
