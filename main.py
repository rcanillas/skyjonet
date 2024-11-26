from typing import List, Dict, Union
from pprint import pprint
from tqdm import tqdm

import os
import json
import random
import copy

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

sns.set_style()
# random.seed(42)

ROW_LENGTH = 4
COLUMN_LENGTH = 3


class Card:
    def __init__(self, value) -> None:
        self.value = value
        self.is_visible = False
        pass

    def __repr__(self) -> str:
        return f"({self.value},{self.is_visible})"


card_proba = {}


class CardDeck:
    def __init__(self) -> None:
        self.stack: List[Card] = []
        for _ in range(0, 5):
            self.stack.append(Card(-2))
            card_proba[-2] = 5 / 150
        for _ in range(0, 15):
            self.stack.append(Card(0))
            card_proba[0] = 15 / 150
        for v in [-1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]:
            for _ in range(0, 10):
                self.stack.append(Card(v))
                card_proba[v] = 10 / 150
        random.shuffle(self.stack)
        self.discard: List[Card] = []

    def reset_deck(self) -> None:
        if len(self.discard) > 0:
            self.stack = self.stack + self.discard
            self.discard = []
        random.shuffle(self.stack)

    def init_round(self) -> None:
        discarded_card = self.draw_card()
        self.discard_card(discarded_card)

    def discard_card(self, card: Card) -> None:
        card.is_visible = True
        self.discard.append(card)

    def draw_card(self, is_visible: bool = False) -> Card:
        drawn_card = self.stack.pop()
        drawn_card.is_visible = is_visible
        return drawn_card

    def get_discarded_card(self) -> Card:
        return self.discard.pop()

    def show_deck(self) -> None:
        print(f"There are {len(self.stack)} cards left in the draw pile.")
        print(
            f"There are {len(self.discard)} cards in the discard pile. The value of the last card is {self.discard[-1].value}"
        )
        return


class Player:
    def __init__(self, player_name: str) -> None:
        self.player_name: str = player_name
        self.card_board = []
        self.last_turn = False
        self.q_draw_action = {}
        self.q_replace_or_reveal_action = {}
        self.q_replace_card_action = {}
        self.q_reveal_card_action = {}
        self.move_history = []
        self.has_won = False
        self.previous_score = None
        self.exploration_prob = 0.5
        self.output_folder = f"{player_name}_data/"
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)
        else:
            with open(
                f"{self.output_folder}draw_action_table.json", "r"
            ) as q_table_draw_action_file:
                old_q_table_draw_action = json.load(q_table_draw_action_file)
                self.q_draw_action = self.q_draw_action | old_q_table_draw_action
            with open(
                f"{self.output_folder}replace_or_reveal_action_table.json", "r"
            ) as q_table_replace_or_reveal_action_file:
                old_q_table_replace_or_reveal_action = json.load(
                    q_table_replace_or_reveal_action_file
                )
                self.q_replace_or_reveal_action = (
                    self.q_replace_or_reveal_action
                    | old_q_table_replace_or_reveal_action
                )
            with open(
                f"{self.output_folder}replace_card_action_table.json", "r"
            ) as q_table_replace_card_action_file:
                old_q_table_replace_card_action = json.load(
                    q_table_replace_card_action_file
                )
                self.q_table_replace_card_action = (
                    self.q_replace_card_action | old_q_table_replace_card_action
                )
            with open(
                f"{self.output_folder}reveal_card_action_table.json", "r"
            ) as q_table_reveal_card_action_file:
                old_q_table_reveal_card_action = json.load(
                    q_table_reveal_card_action_file
                )
                self.q_table_reveal_card_action = (
                    self.q_reveal_card_action | old_q_table_reveal_card_action
                )

    def draw_board(self, card_deck: CardDeck) -> None:
        self.card_board = []
        for _ in range(0, COLUMN_LENGTH):
            col_cards = []
            for _ in range(0, ROW_LENGTH):
                col_cards.append(card_deck.draw_card())
            self.card_board.append(col_cards)
        # pprint(self.replace_card_action_rewards)
        # pprint(self.reveal_card_action_rewards)

    def replace_card(self, card: Card, line: int, col: int) -> List[Card]:
        card_to_discard = copy.copy(self.card_board[line][col])
        self.card_board[line][col] = card
        # print(
        #    f"Player {self.player_name} replaces the card {card_to_discard.value} that was {'visible' if card_to_discard.is_visible else 'hidden'} at [{line},{col}] by a card {card.value}"
        # )
        cards_to_remove = [card_to_discard] + self.check_columns()
        return cards_to_remove

    def compute_visible_score(self, card_board) -> int:
        score = 0
        unknown = 0
        for col in card_board:
            for card in col:
                if card.is_visible:
                    score += card.value
                else:
                    unknown += 1
        # print(
        #    f"Player {self.player_name} score is {score} with {unknown} cards not revealed."
        # )
        return score

    def check_columns(self) -> List[Card]:
        cards_to_remove = []
        for i in range(len(self.card_board[0])):
            card1 = self.card_board[0][i]
            card2 = self.card_board[1][i]
            card3 = self.card_board[2][i]
            if (card1.value == card2.value == card3.value) & (
                card1.is_visible and card2.is_visible and card3.is_visible
            ):
                # print("column with same values found !")
                cards_to_remove = [card1, card2, card3]
        if len(cards_to_remove) > 0:
            self.card_board[0].remove(cards_to_remove[0])
            self.card_board[1].remove(cards_to_remove[1])
            self.card_board[2].remove(cards_to_remove[2])
        return cards_to_remove

    def compute_final_score(self) -> int:
        score = 0
        for col in self.card_board:
            for card in col:
                score += card.value
        # print(f"Player {self.player_name} final score is {score}.")
        return score

    def reveal_card(self, line, col) -> List[Card]:
        cards_to_remove = []
        if not self.card_board[line][col].is_visible:
            self.card_board[line][col].is_visible = True
            # print(
            #    f"Player {self.player_name} reveals the card at [{line},{col}]: it's a {self.card_board[line][col].value} !"
            # )
            cards_to_remove = self.check_columns()
        return cards_to_remove

    def show_board(self) -> None:
        print(f"{self.player_name} cards:")
        for line in self.card_board:
            print(
                f" - ".join(
                    [
                        str(line[col].value) if line[col].is_visible else "x"
                        for col in range(len(line))
                    ]
                )
            )
        return

    def has_hidden_cards(self) -> bool:
        has_hidden_cards = False
        for col in self.card_board:
            for c in col:
                if c.is_visible == False:
                    has_hidden_cards = True
        return has_hidden_cards

    def get_player_cards(self, card_board) -> List[List[Union[int, str]]]:
        return [
            [copy.copy(c.value) if c.is_visible else "X" for c in col]
            for col in card_board
        ]

    def get_environment(
        self,
        drawn_card: Card = None,
        discard_card: Card = None,
        card_board: List[List[Card]] = None,
    ) -> str:
        if not card_board:
            card_board = self.card_board
        return json.dumps(
            {
                "drawn_card": copy.copy(drawn_card.value) if drawn_card else None,
                "discard_card": (
                    (copy.copy(discard_card.value)) if discard_card else None
                ),
                "player_cards": self.get_player_cards(card_board),
            },
        )

    def select_draw_action(self, card_deck) -> str:
        next_move = None
        allowed_moves = ["from_deck", "from_discard"]

        def select_random_move(allowed_moves: List[str]) -> tuple[str, str]:
            next_move = random.choice(allowed_moves)
            if next_move == "from_deck":
                env = None
            else:
                env = self.get_environment(card_deck.discard[-1], None)
                if env not in self.q_draw_action:
                    self.q_draw_action[env] = 0
            return next_move, env

        def select_best_move(
            discarded_card: Card, allowed_moves: List[str]
        ) -> tuple[str, str]:
            card_values = []
            for row in self.card_board:
                for c in row:
                    card_values.append(c.value)
            if discarded_card.value < min(card_values):
                next_move = "from_discard"
            else:
                next_move = "from_deck"
            next_move = random.choice(allowed_moves)
            if next_move == "from_deck":
                env = None
            else:
                env = self.get_environment(card_deck.discard[-1], None)
                if env not in self.q_draw_action:
                    self.q_draw_action[env] = 0
            return next_move, env

        next_move_list = []
        n = np.random.random()
        if n > self.exploration_prob:
            maxG = -10e15
            possible_environment_from_discard = self.get_environment(
                card_deck.discard[-1], None
            )
            next_move_list.append(("from_discard", possible_environment_from_discard))
            for card_value in [
                v for v in range(-2, 13) if v != card_deck.discard[-1].value
            ]:
                possible_environment_from_deck = self.get_environment(
                    Card(card_value), None
                )
                next_move_list.append(("from_deck", possible_environment_from_deck))
            for move, possible_env in next_move_list:
                try:
                    reward = self.q_draw_action[possible_env]
                except KeyError:
                    self.q_draw_action[possible_env] = 0
                    reward = self.q_draw_action[possible_env]
                move_q_probabilized = (
                    reward
                    if move == "from_discard"
                    else reward * card_proba[json.loads(possible_env)["drawn_card"]]
                )
                if move_q_probabilized > maxG:
                    next_move = move
                    env = possible_env
                    maxG = move_q_probabilized
            if maxG == 0:
                next_move, env = select_best_move(card_deck.discard[-1], allowed_moves)

        else:
            next_move, env = select_random_move(allowed_moves)

            def select_random_move(allowed_moves: List[str]) -> tuple[str, str]:
                next_move = random.choice(allowed_moves)
                if next_move == "from_deck":
                    env = None
                else:
                    env = self.get_environment(card_deck.discard[-1], None)
                    if env not in self.q_draw_action:
                        self.q_draw_action[env] = 0
                return next_move, env

        return next_move, env

    def select_replace_or_reveal_action(self, drawn_card: Card) -> tuple[str, str]:
        next_move = None
        next_move_list = []
        allowed_moves = ["replace_card", "reveal_card"]

        def select_random_move(allowed_moves: List[str]) -> tuple[str, str]:
            next_move = random.choice(allowed_moves)
            if next_move == "replace_card":
                env = self.get_environment(drawn_card, None)
            else:
                env = self.get_environment(None, drawn_card)
            if env not in self.q_replace_or_reveal_action:
                self.q_replace_or_reveal_action[env] = 0
            return next_move, env

        def select_best_move(
            self, drawn_card: Card, allowed_moves: List[str]
        ) -> tuple[str, str]:
            next_move = random.choice(allowed_moves)
            cards_values = []
            for row in self.card_board:
                for card in row:
                    if card.is_visible:
                        cards_values.append(card.value)
            if drawn_card.value < min(cards_values):
                next_move = "replace_card"
            else:
                if drawn_card.value not in [-2, -1, 0]:
                    for k in range(len(self.card_board[0])):
                        card1 = self.card_board[0][k]
                        card2 = self.card_board[1][k]
                        card3 = self.card_board[2][k]
                        if (drawn_card.value == card2.value == card3.value) & (
                            card2.is_visible & card3.is_visible
                        ):
                            next_move = "replace_card"
                        elif (drawn_card.value == card1.value == card3.value) & (
                            card1.is_visible & card3.is_visible
                        ):
                            next_move = "replace_card"
                        elif (drawn_card.value == card1.value == card2.value) & (
                            card1.is_visible & card2.is_visible
                        ):
                            next_move = "replace_card"
            if next_move == "replace_card":
                env = self.get_environment(drawn_card, None)
            else:
                env = self.get_environment(None, drawn_card)
            if env not in self.q_replace_or_reveal_action:
                self.q_replace_or_reveal_action[env] = 0
            return next_move, env

        n = np.random.random()

        if n > self.exploration_prob:
            maxG = -10e15
            possible_environment_from_replace = self.get_environment(drawn_card, None)
            next_move_list.append(("replace_card", possible_environment_from_replace))
            possible_environment_from_reveal = self.get_environment(None, drawn_card)
            next_move_list.append(("reveal_card", possible_environment_from_reveal))
            for move, possible_env in next_move_list:
                try:
                    reward = self.q_replace_or_reveal_action[possible_env]
                except KeyError:
                    self.q_replace_or_reveal_action[possible_env] = 0
                    reward = self.q_replace_or_reveal_action[possible_env]
                if reward > maxG:
                    next_move = move
                    env = possible_env
                    maxG = self.q_replace_or_reveal_action[env]
            if maxG == 0:
                next_move, env = select_random_move(allowed_moves)
        else:
            next_move, env = select_random_move(allowed_moves)

        return next_move, env

    def select_card_to_replace(self, drawn_card) -> tuple[int, int]:
        hidden_population = [
            (x, y)
            for x in range(0, COLUMN_LENGTH)
            for y in range(0, len(self.card_board[0]))
            if not self.card_board[x][y].is_visible
        ]
        visible_population = [
            (x, y)
            for x in range(0, COLUMN_LENGTH)
            for y in range(0, len(self.card_board[0]))
            if self.card_board[x][y].is_visible
        ]

        def replace_random_card(
            population: List[tuple[int, int]]
        ) -> tuple[tuple[int, int], str]:
            card_position = random.choice(population)
            temp_board = copy.deepcopy(self.card_board)
            discarded_card = temp_board[card_position[0]][card_position[1]]
            temp_board[card_position[0]][card_position[1]] = drawn_card
            env = self.get_environment(None, discarded_card, temp_board)
            if env not in self.q_replace_card_action:
                self.q_replace_card_action[env] = 0
            return card_position, env

        def replace_best_card(
            drawn_card: Card, population: List[tuple[int, int]]
        ) -> tuple[tuple[int, int], str]:
            card_position = random.choice(population)
            possible_moves = []
            remove_value = -3
            for i, row in enumerate(self.card_board):
                for j, card in enumerate(row):
                    if card.is_visible:
                        if (card.value > drawn_card.value) & (
                            card.value > remove_value
                        ):
                            card_to_replace = (i, j)
                            remove_value = card.value
                            expected_gain = card.value - drawn_card.value
                            possible_moves.append((card_to_replace, expected_gain))
                    else:
                        if drawn_card.value not in [-2, -1, 0]:
                            for k in range(len(self.card_board[0])):
                                card1 = self.card_board[0][k]
                                card2 = self.card_board[1][k]
                                card3 = self.card_board[2][k]
                                expected_gain = drawn_card.value * 3
                                if (drawn_card.value == card2.value == card3.value) & (
                                    card2.is_visible & card3.is_visible
                                ):
                                    possible_moves.append(((0, k), expected_gain))
                                elif (
                                    drawn_card.value == card1.value == card3.value
                                ) & (card1.is_visible & card3.is_visible):
                                    possible_moves.append(((1, k), expected_gain))
                                elif (
                                    drawn_card.value == card1.value == card2.value
                                ) & (card1.is_visible & card2.is_visible):
                                    possible_moves.append(((2, k), expected_gain))
            best_expected_gain = 0
            # pprint(possible_moves)
            for move_card_position, move_expected_gain in possible_moves:
                if move_expected_gain > best_expected_gain:
                    card_position = move_card_position
                    best_expected_gain = move_expected_gain

            temp_board = copy.deepcopy(self.card_board)
            discarded_card = temp_board[card_position[0]][card_position[1]]
            temp_board[card_position[0]][card_position[1]] = drawn_card
            env = self.get_environment(None, discarded_card, temp_board)
            if env not in self.q_replace_card_action:
                self.q_replace_card_action[env] = 0
            return card_position, env

        possible_position_list = []
        n = np.random.random()
        if n > self.exploration_prob:
            maxG = -10e15
            for possible_card_position_visible in visible_population:
                temp_board = copy.deepcopy(self.card_board)
                temp_board[possible_card_position_visible[0]][
                    possible_card_position_visible[1]
                ] = drawn_card
                future_env = self.get_environment(
                    None,
                    self.card_board[possible_card_position_visible[0]][
                        possible_card_position_visible[1]
                    ],
                    temp_board,
                )
                try:
                    possible_position_list.append(
                        (
                            possible_card_position_visible,
                            future_env,
                            (self.q_replace_card_action[future_env]),
                        )
                    )
                except KeyError:
                    self.q_replace_card_action[future_env] = 0
                possible_position_list.append(
                    (
                        possible_card_position_visible,
                        future_env,
                        (self.q_replace_card_action[future_env]),
                    )
                )
            for possible_card_position_hidden in hidden_population:
                temp_board = copy.deepcopy(self.card_board)
                temp_board[possible_card_position_hidden[0]][
                    possible_card_position_hidden[1]
                ] = drawn_card
                for card_value in range(-2, 13):
                    future_env = self.get_environment(
                        None, Card(card_value), temp_board
                    )
                    try:
                        possible_position_list.append(
                            (
                                possible_card_position_hidden,
                                future_env,
                                self.q_replace_card_action[future_env]
                                * card_proba[card_value],
                            )
                        )
                    except KeyError:
                        self.q_replace_card_action[future_env] = 0
                    possible_position_list.append(
                        (
                            possible_card_position_hidden,
                            future_env,
                            self.q_replace_card_action[future_env]
                            * card_proba[card_value],
                        )
                    )
            for possible_card_position, possible_env, reward in possible_position_list:
                # print(reward)
                if reward > maxG:
                    card_position = possible_card_position
                    env = possible_env
                    maxG = reward
            if maxG == 0:
                card_position, env = replace_best_card(
                    drawn_card, hidden_population + visible_population
                )
        else:
            card_position, env = replace_random_card(
                hidden_population + visible_population
            )
        return card_position, env

    def select_card_to_reveal(self) -> tuple[tuple[int, int], str]:
        population = [
            (x, y)
            for x in range(0, COLUMN_LENGTH)
            for y in range(0, len(self.card_board[0]))
            if not self.card_board[x][y].is_visible
        ]

        def reveal_random_card(
            population: List[tuple[int, int]]
        ) -> tuple[tuple[int, int], str]:
            card_position = random.choice(population)
            temp_board = copy.deepcopy(self.card_board)
            temp_board[card_position[0]][card_position[1]].is_visible = True
            env = self.get_environment(None, None, temp_board)
            if env not in self.q_reveal_card_action:
                self.q_reveal_card_action[env] = 0
            return card_position, env

        def reveal_missing_card(
            population: List[tuple[int, int]]
        ) -> tuple[tuple[int, int], str]:
            possible_card_list = []
            for i in range(len(self.card_board[0])):
                card1 = self.card_board[0][i]
                card2 = self.card_board[1][i]
                card3 = self.card_board[2][i]
                if (not card1.is_visible) & card2.is_visible & card3.is_visible:
                    possible_card_list.append((0, i))
                elif card1.is_visible & (not card2.is_visible) & card3.is_visible:
                    possible_card_list.append((1, i))
                elif card1.is_visible & card2.is_visible & (not card3.is_visible):
                    possible_card_list.append((2, i))
            if possible_card_list:
                card_position = random.choice(possible_card_list)
            else:
                card_position = random.choice(population)
            temp_board = copy.deepcopy(self.card_board)
            temp_board[card_position[0]][card_position[1]].is_visible = True
            env = self.get_environment(None, None, temp_board)
            if env not in self.q_reveal_card_action:
                self.q_reveal_card_action[env] = 0
            return card_position, env

        possible_position_list = []
        n = np.random.random()
        if n > self.exploration_prob:
            for possible_card_position in population:
                maxG = -10e15
                for card_value in range(-2, 13):
                    temp_board = copy.deepcopy(self.card_board)
                    temp_board[possible_card_position[0]][possible_card_position[1]] = (
                        Card(card_value)
                    )
                    temp_board[possible_card_position[0]][
                        possible_card_position[1]
                    ].is_visible = True
                    future_env = self.get_environment(None, None, temp_board)
                    if future_env not in self.q_reveal_card_action:
                        self.q_reveal_card_action[future_env] = 0
                    possible_position_list.append(
                        (
                            possible_card_position,
                            future_env,
                            (
                                self.q_reveal_card_action[future_env]
                                * card_proba[card_value]
                            ),
                        )
                    )
            for possible_card_position, possible_env, reward in possible_position_list:
                if reward > maxG:
                    card_position = possible_card_position
                    env = possible_env
                    maxG = reward
            if reward == 0:
                card_position, env = reveal_missing_card(population)
        else:
            card_position, env = reveal_random_card(population)

        return card_position, env

    def init_game(self) -> None:
        # Player reveals two cards from its board at the beginning of the game
        first_card_revealed, environment1 = self.select_card_to_reveal()
        self.reveal_card(first_card_revealed[0], first_card_revealed[1])
        self.move_history.append(
            {
                "type": "first_card_reveal_move",
                "environment": environment1,
                "actions": [(first_card_revealed, environment1)],
            }
        )
        # Revealing the second card
        second_card_revealed, environment2 = self.select_card_to_reveal()
        self.reveal_card(second_card_revealed[0], second_card_revealed[1])
        self.move_history.append(
            {
                "type": "second_card_reveal_move",
                "environment": environment2,
                "actions": [
                    (second_card_revealed, environment2),
                ],
            }
        )
        # pprint(self.move_history)
        return

    def reset_board(self, card_deck: CardDeck) -> None:
        for row in self.card_board:
            for card in row:
                card_deck.discard_card(card)
        self.last_turn = False

    def play_turn(self, card_deck: CardDeck) -> None:
        # Player either draw from draw pile or take card from discard pile
        actions = []
        draw_action, environment = self.select_draw_action(card_deck)
        if draw_action == "from_discard":
            # If the card is drawn from the discard pile, Player has to exchange the card with one on the board.
            actions.append((draw_action, environment))
            drawn_card = card_deck.get_discarded_card()
            card_to_replace, environment = self.select_card_to_replace(drawn_card)
            actions.append((card_to_replace, environment))
            replaced_cards = self.replace_card(
                drawn_card, card_to_replace[0], card_to_replace[1]
            )
            self.move_history.append(
                {
                    "type": "replace_from_discard_move",
                    "actions": actions,
                }
            )

        elif draw_action == "from_deck":
            # If the card is drawn from the deck pile, Player can exchange the card with one on the board or discard the card and reveal a card on the board.
            drawn_card = card_deck.draw_card(is_visible=True)
            actions.append((draw_action, self.get_environment(drawn_card)))
            if self.get_environment(drawn_card) not in self.q_draw_action:
                self.q_draw_action[self.get_environment(drawn_card)] = 0
            replace_or_reveal_action, environment = (
                self.select_replace_or_reveal_action(drawn_card)
            )
            actions.append((replace_or_reveal_action, environment))
            if replace_or_reveal_action == "replace_card":
                card_to_replace, environment = self.select_card_to_replace(drawn_card)
                actions.append((card_to_replace, environment))
                replaced_cards = self.replace_card(
                    drawn_card, card_to_replace[0], card_to_replace[1]
                )
                self.move_history.append(
                    {
                        "type": "replace_from_deck_move",
                        "actions": actions,
                    }
                )
            elif replace_or_reveal_action == "reveal_card":
                card_to_reveal, environment = self.select_card_to_reveal()
                actions.append((card_to_reveal, environment))
                card_deck.discard_card(drawn_card)
                replaced_cards = self.reveal_card(card_to_reveal[0], card_to_reveal[1])
                self.move_history.append(
                    {"type": "reveal_from_deck_move", "actions": actions}
                )

        for replaced_card in replaced_cards:
            card_deck.discard_card(replaced_card)
        return

    def update_q_tables(self, alpha=0.1) -> None:
        if not self.previous_score:
            target = 1 if self.has_won else -1
        else:
            final_score = self.compute_final_score()
            target = 1 if final_score < self.previous_score else -1
            self.previous_score = final_score
        for move in reversed(self.move_history):
            # print(move["type"])
            for action, env in move["actions"]:
                # print(action)
                # print(env)
                if action == "from_discard" or action == "from_deck":
                    # pprint(env)
                    # pprint(self.q_table_draw_action.keys())
                    self.q_draw_action[env] = self.q_draw_action[env] + alpha * (
                        target - self.q_draw_action[env]
                    )
                elif action == "replace_card" or action == "reveal_card":
                    # pprint(env)
                    # pprint(self.q_table_replace_or_reveal_action.keys())
                    self.q_replace_or_reveal_action[env] = (
                        self.q_replace_or_reveal_action[env]
                        + alpha * (target - self.q_replace_or_reveal_action[env])
                    )
                else:
                    if (
                        move["type"] == "replace_from_discard_move"
                        or move["type"] == "replace_from_deck_move"
                    ):
                        # pprint(env)
                        # pprint(self.q_replace_card_action.keys())
                        self.q_replace_card_action[env] = self.q_replace_card_action[
                            env
                        ] + alpha * (target - self.q_replace_card_action[env])
                    elif move["type"] == "reveal_from_deck_move":
                        # pprint(env)
                        # pprint(self.q_reveal_card_action.keys())
                        self.q_reveal_card_action[env] = self.q_reveal_card_action[
                            env
                        ] + alpha * (target - self.q_reveal_card_action[env])
            # print("OK")
        with open(
            f"{self.output_folder}draw_action_table.json", "w"
        ) as q_table_draw_action_file:
            json.dump(
                self.q_draw_action,
                q_table_draw_action_file,
            )
        with open(
            f"{self.output_folder}replace_or_reveal_action_table.json", "w"
        ) as q_table_replace_or_reveal_action_file:
            json.dump(
                self.q_replace_or_reveal_action,
                q_table_replace_or_reveal_action_file,
            )
        with open(
            f"{self.output_folder}replace_card_action_table.json", "w"
        ) as q_table_replace_card_action_file:
            json.dump(self.q_replace_card_action, q_table_replace_card_action_file)
        with open(
            f"{self.output_folder}reveal_card_action_table.json", "w"
        ) as q_table_reveal_card_action_file:
            json.dump(self.q_reveal_card_action, q_table_reveal_card_action_file)

        self.exploration_prob -= 10e-5
        self.move_history = []


def play_game(card_deck: CardDeck, player_list: List[Player], i: int) -> None:
    # print("Starting game", i)
    card_deck.reset_deck()
    for player in player_list:
        player.draw_board(card_deck)
        player.init_game()
        # player.show_board()
    card_deck.init_round()
    i = 1
    player_ordered_turn = sorted(
        player_list, key=lambda p: p.compute_visible_score(p.card_board)
    )
    round_over = False
    while not round_over:
        # print(f"__ Turn {i} ___")
        # player.show_board()
        for player in player_ordered_turn:
            if player.last_turn:
                round_over = True
                break
            player.play_turn(card_deck)
            # player.show_board()
            # card_deck.show_deck()
            player.last_turn = not player.has_hidden_cards()
            if player.last_turn:
                # print(
                #    f"{player.player_name} has revealed all his board ! This is the last turn !"
                # )
                pass
        i += 1
    # print("Round over !")
    min_score = 1000
    best_player = None
    for player in player_ordered_turn:
        player_final_score = player.compute_final_score()
        if player_final_score < min_score:
            min_score = player_final_score
            best_player = player
        player.reset_board(card_deck)
    best_player.has_won = True

    # print(
    #    f"The winner of the round is {best_player.player_name} with a score of {min_score} !"
    # )
    # pprint(player1.move_history)
    # pprint(len(player2.move_history))


if __name__ == "__main__":
    card_deck = CardDeck()
    player1 = Player("Random Tom")
    player2 = Player("Smarty Lucy")
    player3 = Player("Average Joe")
    player1.exploration_prob = 0.9
    player2.exploration_prob = 0.1
    player3.exploration_prob = 0.5
    player_scores = []
    player_list = [player1, player2, player3]
    for i in tqdm(range(2000)):
        play_game(card_deck, player_list, i)
        for player in player_list:
            player.update_q_tables()
            player_scores.append(
                {
                    "game": i,
                    "score": player.compute_final_score(),
                    "player": player.player_name,
                }
            )
    scores_df = pd.DataFrame(player_scores)
    fig = plt.figure(figsize=(50, 7))
    ax = sns.lineplot(data=scores_df, x="game", y="score", hue="player", marker="o")
    color_list = ["blue", "orange", "green"]
    for i, player in enumerate(player_list):
        ax1 = plt.axhline(
            scores_df.loc[scores_df["player"] == player.player_name]["score"].mean(),
            ls="--",
            c=color_list[i],
        )

    plt.show()
