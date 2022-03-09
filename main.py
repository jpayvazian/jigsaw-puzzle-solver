import sift
import segmentation

def prompt(str):
    return input(str).lower() == "y"

if __name__ == "__main__":
    if prompt("Resize images? [y/n] "):
        segmentation.resize_images("Puzzle Examples/12pieces.jpg", "Puzzle Examples/12puzzle.jpg")

    if prompt("Segment pieces? [y/n] "):
        segmentation.segment_pieces("Puzzle Examples/12pieces.jpg", 20, 40, 10)

    if prompt("Preform SIFT matching? [y/n] "):
        sift.sift_demo("Puzzle Examples/12puzzle.jpg", 40, 10)
