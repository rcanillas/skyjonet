from flask import Flask, request, jsonify
from flask_cors import CORS


app = Flask(__name__)
CORS(app)
game_instances = {}


@app.route("/create_debate", methods=["GET"])
def create_debate():
    session_id = debate_manager.create_debate()
    return jsonify({"session_id": session_id})


@app.route("/cartes", methods=["GET"])
def debate():
    result = {
        "boardCards": [{"value": "A", "hidden": True}],
        "players": [
            [{"value": "1", "hidden": True}],
            [{"value": "2", "hidden": False}],
        ],
        "pile": [{"value": "5", "hidden": True}],
        "discard": [
            {"value": "K", "hidden": False},
        ],
    }

    return jsonify(result)


if __name__ == "__main__":
    app.run(port=5000, debug=True)
