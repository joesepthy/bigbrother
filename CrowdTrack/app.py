# -*- coding: utf-8 -*-
from flask import Flask, render_template

app = Flask(__name__)

# cam1, cam2 블루프린트 import
from cam1 import cam1_bp
from cam2 import cam2_bp

# 블루프린트 등록
app.register_blueprint(cam1_bp, url_prefix='/cam1')
app.register_blueprint(cam2_bp, url_prefix='/cam2')

@app.route('/')
def index():
    return render_template('main.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)