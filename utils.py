import os
import shutil

def make_dir(name):
    try:
        # Try to create folder locally
        os.mkdir(name)
    except FileExistsError:
        # If folder already exists, wipe it & remake it
        if os.path.exists(name):
            shutil.rmtree(name)
        os.mkdir(name)

def dir_size(path):
    return len(os.listdir(path))

def save_coordinates(num, x, y):
    with open("Contours/coordinates.txt", "a") as f:
        f.writelines("{} {} {} \n".format(num, x, y))

def save_grid(num, x, y, xmin, xmax, ymin, ymax):
    with open("Contours/grid.txt", "a") as f:
        f.writelines("{} {} {} {} {} {} {} \n".format(num, x, y, xmin, xmax, ymin, ymax))

def find_coordinates(num):
    with open("Contours/coordinates.txt", "r") as f:
        lines = f.readlines()
        for line in lines:
            line_list = line.split(" ")
            if int(line_list[0]) == num:
                return line_list[1:3]


def find_grid(num):
    with open("Contours/grid.txt", "r") as f:
        lines = f.readlines()
        for line in lines:
            line_list = line.split(" ")
            if int(line_list[0]) == num:
                return line_list[1:7]