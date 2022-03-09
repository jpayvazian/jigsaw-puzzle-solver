import os
from flask import Flask, render_template, request, send_from_directory, session
import segmentation
import utils
import sift

app = Flask(__name__)
app.secret_key = os.urandom(24)
@app.route("/")
def index():
    return render_template('index.html')

@app.route("/Puzzle Mask/<path>")
def send_mask_file(path):
    return send_from_directory('Puzzle Mask', path)

@app.route("/SIFT Sorted/<path>")
def send_sift_file(path):
    return send_from_directory('SIFT Sorted', path)

@app.route("/submit", methods=['POST'])
def submit_fcn():
    if request.method == "POST":
        session['NUM_PIECES'] = int(request.form['NumPieces'])
        img_puzzle = request.files['Puzzle']
        img_pieces = request.files['Pieces']
        if img_puzzle and img_pieces:
            img_puzzle.save("Puzzle Examples/ui_full_img.jpg")
            img_pieces.save("Puzzle Examples/ui_pieces_img.jpg")
            segmentation.adjust_mask("Puzzle Examples/ui_pieces_img.jpg", 40, 15)
            return render_template('mask.html', bgr_initial=40, kernel_initial=15)

@app.route("/mask", methods=['POST'])
def mask_fcn():
    if request.method == "POST":
        bgr = int(request.form["SliderBGR"])
        kernel = int(request.form["SliderKernel"])
        segmentation.adjust_mask("Puzzle Examples/ui_pieces_img.jpg", bgr, kernel)
        return render_template('mask.html', bgr_initial=bgr, kernel_initial=kernel)

@app.route("/sift", methods=['POST'])
def sift_fcn():
    if request.method == "POST":
        bgr = int(request.form["SliderBGR"])
        kernel = int(request.form["SliderKernel"])
        #segmentation.resize_images("Puzzle Examples/ui_full_img.jpg", "Puzzle Examples/ui_pieces_img.jpg")
        segmentation.resize_images("Puzzle Examples/ui_pieces_img.jpg", "Puzzle Examples/ui_full_img.jpg")
        segmentation.segment_pieces("Puzzle Examples/ui_pieces_img.jpg", session.get('NUM_PIECES', None), bgr, kernel)
        matches_found = sift.sift_demo("Puzzle Examples/ui_full_img.jpg", bgr, kernel)
        return render_template('sift.html', size=utils.dir_size("Pieces"), matches=matches_found)

if __name__ == "__main__":
    app.run(debug=True)
