from flask import Flask, request, jsonify
from flask_cors import CORS
import json

from main import Game


app = Flask(__name__)
CORS(app)
game_instances = {}
game_id = 1


@app.route("/init_game", methods=["POST"])
def init_game():
    global game_id
    request_data = request.get_json()
    print(request_data)
    player_list = request_data["player_list"]
    game_instances[game_id] = Game(game_id, player_list)
    game_state = game_instances[game_id].export_game_state()
    game_id += 1
    return game_state


@app.route("/<int:game_id>/reveal_card", methods=["GET"])
def reveal_card(game_id: int):
    player_name = request.args.get("playerName")
    card_column = request.args.get("cardX")
    card_line = request.args.get("cardY")
    selected_game = game_instances[game_id]
    selected_game.reveal_card(player_name, card_line, card_column)
    return selected_game.export_game_state()


@app.route("/<int:game_id>/draw_card", methods=["GET"])
def draw_card(game_id: int):
    selected_game = game_instances[game_id]
    selected_game.draw_card()
    return selected_game.export_game_state()


@app.route("/<int:game_id>/get_discard_card", methods=["GET"])
def get_discard_card(game_id: int):
    selected_game = game_instances[game_id]
    selected_game.get_discard_card()
    return selected_game.export_game_state()


if __name__ == "__main__":
    app.run(port=5000, debug=True)
