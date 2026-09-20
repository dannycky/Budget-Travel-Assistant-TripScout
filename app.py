"""
Travel Assistant Chatbot - Flask Web Server
===========================================
Serves templates/index.html as the frontend and exposes a POST /chat
endpoint backed by the RAG TravelChatbot from chatbot.py.

Usage:
    python app.py
    Then open http://127.0.0.1:5001 in your browser.
"""

from flask import Flask, render_template, request, jsonify

from chatbot import TravelChatbot

app = Flask(__name__)

# Initialize the RAG chatbot once at startup
bot = TravelChatbot()


@app.route("/")
def index():
    """Serve the HTML frontend."""
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():
    """Handle a chat message from the frontend and return the answer."""
    data = request.get_json(silent=True) or {}
    message = (data.get("message") or "").strip()
    if not message:
        return jsonify({"error": "No message provided."}), 400
    try:
        answer = bot.ask(message)
        return jsonify({"answer": answer})
    except Exception as e:
        return jsonify({"error": f"Something went wrong: {e}"}), 500


if __name__ == "__main__":
    # Port 5001 because macOS AirPlay Receiver occupies 5000 by default
    app.run(host="127.0.0.1", port=5001, debug=True)
