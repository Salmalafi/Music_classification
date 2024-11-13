from flask import Flask, render_template

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')  # Serve the index.html page

if __name__ == '__main__':
    app.run(debug=True, port=5001)  # Run the app on port 5001
