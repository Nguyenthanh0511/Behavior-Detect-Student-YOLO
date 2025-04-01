from flask import Flask, render_template, Response, jsonify
import cv2

app = Flask(__name__)

# Danh sách đường dẫn video
VIDEO_PATHS = ['VIDEO_PATH1.mp4', 'VIDEO_PATH2.mp4']
current_video_index = 0

# Hàm tạo luồng video
def generate_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Không thể mở video: {video_path}")
        return

    while True:
        success, frame = cap.read()
        if not success:
            break

        # Mã hóa frame sang JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        # Trả về frame theo định dạng multipart
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    cap.release()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(VIDEO_PATHS[current_video_index]),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/next_video', methods=['POST'])
def next_video():
    global current_video_index
    current_video_index = (current_video_index + 1) % len(VIDEO_PATHS)
    return jsonify({
        'message': 'Chuyển sang video tiếp theo',
        'current_index': current_video_index
    })

if __name__ == '__main__':
    app.run(debug=True)
