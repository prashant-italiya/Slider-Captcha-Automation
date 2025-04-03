from flask import Flask, jsonify
import subprocess
import os

app = Flask(__name__)

@app.route('/run_python_script', methods=['GET'])
def run_script():
    """
    Endpoint to trigger the slider captcha solver script.
    Returns a JSON response indicating success or failure.
    """
    try:
        script_path = "slider_captcha_solver.py"
        if not os.path.exists(script_path):
            return jsonify({
                "status": "error",
                "message": f"Script not found: {script_path}"
            }), 404

        # Run the script in a new process
        process = subprocess.Popen(["python", script_path])
        
        return jsonify({
            "status": "success",
            "message": "Slider captcha solver started successfully",
            "pid": process.pid
        }), 200
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Failed to start script: {str(e)}"
        }), 500

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)