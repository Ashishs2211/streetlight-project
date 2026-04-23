from flask import Flask, render_template, request, redirect, session, url_for, send_from_directory, Response
import sqlite3
import os
import subprocess
import glob
import shutil
import cv2
from datetime import datetime

app = Flask(__name__)
app.secret_key = "streetlight_secret_key"

UPLOAD_FOLDER = "static/uploads"
RESULT_FOLDER = "static/results"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

DB_NAME = "streetlight.db"
MODEL_PATH = "runs/detect/models/streetlight_yolov85/weights/best.pt"
if os.environ.get("RENDER"):
    camera = None
else:
    camera = cv2.VideoCapture(0)
def get_stats():

    conn = sqlite3.connect(DB_NAME)
    cur = conn.cursor()

    cur.execute("SELECT COUNT(*) FROM detections")
    total_images = cur.fetchone()[0]

    cur.execute("SELECT COALESCE(SUM(faults),0) FROM detections")
    total_faults = cur.fetchone()[0]

    today = datetime.now().strftime("%d-%m-%Y")

    cur.execute(
        "SELECT COUNT(*) FROM detections WHERE created_at LIKE ?",
        (today + "%",)
    )

    today_uploads = cur.fetchone()[0]

    conn.close()

    return total_images, total_faults, today_uploads


# ======================
# DATABASE
# ======================
def init_db():
    conn = sqlite3.connect(DB_NAME)
    cur = conn.cursor()

    cur.execute("""
    CREATE TABLE IF NOT EXISTS users(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT,
        email TEXT,
        password TEXT
    )
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS detections(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        image_name TEXT,
        faults INTEGER,
        created_at TEXT
    )
    """)

    conn.commit()
    conn.close()

init_db()


# ======================
# LOGIN REQUIRED
# ======================
def logged_in():
    return "user" in session


# ======================
# REGISTER
# ======================
@app.route("/register", methods=["GET","POST"])
def register():

    if request.method == "POST":

        username = request.form["username"]
        email = request.form["email"]
        password = request.form["password"]

        conn = sqlite3.connect(DB_NAME)
        cur = conn.cursor()

        cur.execute("""
        INSERT INTO users(username,email,password)
        VALUES(?,?,?)
        """,(username,email,password))

        conn.commit()
        conn.close()

        return redirect("/login")

    return render_template("register.html")


# ======================
# LOGIN
# ======================
@app.route("/login", methods=["GET","POST"])
def login():

    if request.method == "POST":

        email = request.form["email"]
        password = request.form["password"]

        conn = sqlite3.connect(DB_NAME)
        cur = conn.cursor()

        cur.execute("""
        SELECT * FROM users
        WHERE email=? AND password=?
        """,(email,password))

        user = cur.fetchone()

        conn.close()

        if user:
            session["user"] = user[1]
            return redirect("/")

    return render_template("login.html")


# ======================
# LOGOUT
# ======================
@app.route("/logout")
def logout():
    session.clear()
    return redirect("/login")


# ======================
# HOME
# ======================
@app.route("/")
def home():

    if not logged_in():
        return redirect("/login")

    total_images, total_faults, today_uploads = get_stats()

    return render_template(
        "index.html",
        username=session["user"],
        total_images=total_images,
        total_faults=total_faults,
        today_uploads=today_uploads
    )
# ======================
# HISTORY
# ======================
@app.route("/history")
def history():

    if not logged_in():
        return redirect("/login")

    conn = sqlite3.connect(DB_NAME)
    cur = conn.cursor()

    cur.execute("SELECT * FROM detections ORDER BY id DESC")
    rows = cur.fetchall()

    conn.close()

    return render_template("history.html", rows=rows)

# ======================
# PROFILE PAGE
# ======================
@app.route("/profile")
def profile():

    if not logged_in():
        return redirect("/login")

    conn = sqlite3.connect(DB_NAME)
    cur = conn.cursor()

    cur.execute("SELECT * FROM users WHERE username=?", (session["user"],))
    user = cur.fetchone()

    conn.close()

    return render_template("profile.html", user=user)


# ======================
# ADMIN PANEL
# ======================
@app.route("/admin")
def admin():

    if not logged_in():
        return redirect("/login")

    conn = sqlite3.connect(DB_NAME)
    cur = conn.cursor()

    cur.execute("SELECT COUNT(*) FROM users")
    total_users = cur.fetchone()[0]

    cur.execute("SELECT COUNT(*) FROM detections")
    total_detections = cur.fetchone()[0]

    cur.execute("SELECT COALESCE(SUM(faults),0) FROM detections")
    total_faults = cur.fetchone()[0]

    conn.close()

    return render_template(
        "admin.html",
        total_users=total_users,
        total_detections=total_detections,
        total_faults=total_faults
    )
# ======================
# CAMERA STREAM
# ======================
def generate_frames():
    while True:
        success, frame = camera.read()

        if not success:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route("/camera")
def camera_page():

    if not logged_in():
        return redirect("/login")

    return render_template("camera.html")


@app.route("/video_feed")
def video_feed():

    if camera is None:
        return "Camera not available on cloud server"

    return Response(
        generate_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )
# ======================
# CAMERA CAPTURE DETECT
# ======================
@app.route("/capture_detect", methods=["POST"])
def capture_detect():

    if camera is None:
        return "Camera not available on cloud server"

    success, frame = camera.read()

    if not success:
        return "Camera Error"

    filename = "camera_capture.jpg"
    filepath = os.path.join(UPLOAD_FOLDER, filename)

    cv2.imwrite(filepath, frame)

    # Remove old outputs
    old = glob.glob("outputs/annotated_images/*")
    for f in old:
        try:
            os.remove(f)
        except:
            pass

    # Run detection
    cmd = [
        "python",
        "detect_image.py",
        "--source",
        filepath,
        "--weights",
        MODEL_PATH
    ]

    subprocess.run(cmd)

    generated = glob.glob("outputs/annotated_images/*")

    if generated:

        latest = generated[0]

        result_path = os.path.join(
            RESULT_FOLDER,
            "camera_capture.jpg"
        )

        shutil.copy(latest, result_path)

        return redirect("/camera_result")

    else:
     shutil.copy(
        filepath,
        os.path.join(RESULT_FOLDER, "camera_capture.jpg")
    )
    return redirect("/camera_result")
# ======================
# DETECT
# ======================
@app.route("/detect", methods=["POST"])
def detect():

    if not logged_in():
        return redirect("/login")

    files = request.files.getlist("file")

    if not files:
        return redirect("/")

    results = []

    for file in files:

        if file.filename == "":
            continue

        filename = file.filename
        filepath = os.path.join(UPLOAD_FOLDER, filename)

        file.save(filepath)

        # Clear previous outputs
        old = glob.glob("outputs/annotated_images/*")
        for f in old:
            try:
                os.remove(f)
            except:
                pass

        # REAL AI Detection
        cmd = [
          "python",
          "detect_image.py",
          "--source",
          filepath,
          "--weights",
          MODEL_PATH,
          "--conf",
          "0.15"
        ]

        subprocess.run(cmd)

        generated = glob.glob("outputs/annotated_images/*")

        if generated:
            latest = generated[0]

            shutil.copy(
                latest,
                os.path.join(RESULT_FOLDER, filename)
            )

        faults = 1

        conn = sqlite3.connect(DB_NAME)
        cur = conn.cursor()

        cur.execute("""
        INSERT INTO detections(image_name,faults,created_at)
        VALUES(?,?,?)
        """, (
            filename,
            faults,
            datetime.now().strftime("%d-%m-%Y %H:%M:%S")
        ))

        conn.commit()
        conn.close()

        results.append({
            "name": filename,
            "faults": faults
        })

    total_images, total_faults, today_uploads = get_stats()

    return render_template(
     "index.html",
     username=session["user"],
     results=results,
     total_images=total_images,
     total_faults=total_faults,
     today_uploads=today_uploads
)
    
# ======================
# DOWNLOAD RESULT
# ======================
@app.route("/download/<filename>")
def download(filename):

    if not logged_in():
        return redirect("/login")

    return send_from_directory(
        RESULT_FOLDER,
        filename,
        as_attachment=True
    )
@app.route("/camera_result")
def camera_result():

    if not logged_in():
        return redirect("/login")

    return render_template("camera_result.html")
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)