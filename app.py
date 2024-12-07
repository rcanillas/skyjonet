from flask import Flask, request, jsonify
from flask_cors import CORS

from main import Game


app = Flask(__name__)
CORS(app)
game_instances = {}
game_id = 1


@app.route("/init_game", methods=["POST"])
def init_game():
    request_data = request.get_json()
    player_list = request_data["player_list"]
    game_instances[game_id] = Game(id, player_list)
    game_id += 1


@app.route("/<int:game_id>/show_game", methods=["GET"])
def draw_card(game_id: int):
    card = 1
    return


@app.route("/<int:game_id>/draw_card", methods=["GET"])
def draw_card(game_id: int):
    card = 1
    return


if __name__ == "__main__":
    app.run(port=5000, debug=True)
