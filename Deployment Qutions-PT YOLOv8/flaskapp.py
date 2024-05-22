from flask import Flask, render_template, Response, request, session
from flask_wtf import FlaskForm
from wtforms import FileField, SubmitField
from werkzeug.utils import secure_filename
from wtforms.validators import InputRequired
import os
import cv2
from YOLO import video_detection, detect_descriptions

app = Flask(__name__)
app.config['SECRET_KEY'] = 'ricoprediansyah'
app.config['UPLOAD_FOLDER'] = 'static/files'

class UploadFileForm(FlaskForm):
    file = FileField("File", validators=[InputRequired()])
    submit = SubmitField("Run")

def generate_frames(path_x='', descriptions=[]):
    yolo_output = video_detection(path_x)
    for detection_ in yolo_output:
        ref, buffer = cv2.imencode('.jpg', detection_)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/', methods=['GET', 'POST'])
@app.route('/home', methods=['GET', 'POST'])
def home():
    session.clear()
    return render_template('indexproject.html')

@app.route("/camera", methods=['GET', 'POST'])
def camera():
    session.clear()
    return render_template('camera.html')

@app.route('/fotovideo', methods=['GET', 'POST'])
def fotovideo():
    form = UploadFileForm()
    description = "Deskripsi tidak tersedia"
    if form.validate_on_submit():
        file = form.file.data
        file.save(os.path.join(os.path.abspath(os.path.dirname(__file__)), app.config['UPLOAD_FOLDER'],
                               secure_filename(file.filename)))
        session['video_path'] = os.path.join(os.path.abspath(os.path.dirname(__file__)), app.config['UPLOAD_FOLDER'],
                                             secure_filename(file.filename))
        descriptions = detect_descriptions(session['video_path'])
        session['descriptions'] = descriptions
        description = ", ".join(descriptions)

    return render_template('fotovideo.html', form=form, description=description)

@app.route('/video')
def video():
    return Response(generate_frames(path_x=session.get('video_path', None), descriptions=session.get('descriptions', [])),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/webapp')
def webapp():
    return Response(generate_frames_web(path_x=0), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)
