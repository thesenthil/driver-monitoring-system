from flask import Flask, jsonify

app = Flask(__name__)

# Global status
status = {"state": "SAFE"}

@app.route("/")
def home():
    return "Driver Monitoring System Running"

@app.route("/status")
def get_status():
    return jsonify(status)

# Function to update status from main.py
def update_status(new_state):
    status["state"] = new_state

if __name__ == "__main__":
    app.run(debug=True)
    