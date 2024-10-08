from typing import List, Dict, Union
from pprint import pprint

import json
import random
import copy


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


class CardDeck:
    def __init__(self) -> None:
        self.stack: List[Card] = []
        for _ in range(0, 5):
            self.stack.append(Card(-2))
        for _ in range(0, 15):
            self.stack.append(Card(0))
        for v in [-1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]:
            for _ in range(0, 10):
                self.stack.append(Card(v))
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


class Player:
    def __init__(self, player_name: str) -> None:
        self.player_name: str = player_name
        self.card_board = []
        self.last_turn = False
        self.draw_action_proba = {}
        self.replace_or_reveal_action_proba = {}
        self.replace_card_action_proba = {}
        self.reveal_card_action_proba = {}
        self.move_history = []
        self.has_won = False

    def draw_board(self, card_deck: CardDeck) -> None:
        self.card_board = []
        for row in range(0, COLUMN_LENGTH):
            col_cards = []
            for col in range(0, ROW_LENGTH):
                col_cards.append(card_deck.draw_card())
                self.replace_card_action_proba[(row, col)] = 1 / (
                    ROW_LENGTH * COLUMN_LENGTH
                )
                self.reveal_card_action_proba[(row, col)] = 1 / (
                    ROW_LENGTH * COLUMN_LENGTH
                )
            self.card_board.append(col_cards)
        # pprint(self.replace_card_action_rewards)
        # pprint(self.reveal_card_action_rewards)

    def replace_card(self, card: Card, line: int, col: int) -> List[Card]:
        card_to_discard = copy.copy(self.card_board[line][col])
        self.card_board[line][col] = card
        print(
            f"Player {self.player_name} replaces the card {card_to_discard.value} that was {'visible' if card_to_discard.is_visible else 'hidden'} at [{line},{col}] by a card {card.value}"
        )
        cards_to_remove = [card_to_discard] + self.check_columns()
        return cards_to_remove

    def compute_current_score(self) -> int:
        score = 0
        unknown = 0
        for col in self.card_board:
            for card in col:
                if card.is_visible:
                    score += card.value
                else:
                    unknown += 1
        print(
            f"Player {self.player_name} score is {score} with {unknown} cards not revealed."
        )
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
                print("column with same values found !")
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
        print(f"Player {self.player_name} final score is {score}.")
        return score

    def reveal_card(self, line, col) -> List[Card]:
        cards_to_remove = []
        if not self.card_board[line][col].is_visible:
            self.card_board[line][col].is_visible = True
            print(
                f"Player {self.player_name} reveals the card at [{line},{col}]: it's a {self.card_board[line][col].value} !"
            )
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

    def has_hidden_cards(self) -> bool:
        has_hidden_cards = False
        for col in self.card_board:
            for c in col:
                if c.is_visible == False:
                    has_hidden_cards = True
        return has_hidden_cards

    def get_player_cards(self) -> List[List[Union[int, str]]]:
        return [
            [copy.copy(c.value) if c.is_visible else "X" for c in col]
            for col in self.card_board
        ]

    def get_environment(self, drawn_card: Card, card_deck: CardDeck) -> str:
        return json.dumps(
            {
                "drawn_card": copy.copy(drawn_card.value) if drawn_card else None,
                "discard_card": (
                    copy.copy(card_deck.discard[-1].value) if card_deck else None
                ),
                "player_cards": self.get_player_cards(),
                "player_score": self.compute_current_score(),
            },
        )

    def select_draw_action(self, environement) -> str:
        if environement not in (self.draw_action_proba.keys()):
            self.draw_action_proba[environement] = [0.50, 0.50]
        return random.choices(
            ["from_deck", "from_discard"], self.draw_action_proba[environement]
        )[0]

    def select_card_to_replace(self, environment) -> tuple[int, int]:
        population = [
            (x, y)
            for x in range(0, COLUMN_LENGTH)
            for y in range(0, len(self.card_board[0]))
        ]
        if environment not in (self.replace_card_action_proba.keys()):
            self.replace_card_action_proba[environment] = [
                1 / (COLUMN_LENGTH * len(self.card_board[0]))
                for _ in range(0, COLUMN_LENGTH * len(self.card_board[0]))
            ]
        card_position = random.choices(
            population, self.replace_card_action_proba[environment]
        )
        return card_position[0]

    def select_replace_or_reveal_action(self, environment) -> str:
        if environment not in self.replace_or_reveal_action_proba:
            self.replace_or_reveal_action_proba[environment] = [0.50, 0.50]
        return random.choices(
            ["replace_card", "reveal_card"],
            self.replace_or_reveal_action_proba[environment],
        )[0]

    def select_card_to_reveal(self, environment) -> tuple[int, int]:
        population = [
            (x, y)
            for x in range(0, COLUMN_LENGTH)
            for y in range(0, len(self.card_board[0]))
        ]
        if environment not in (self.reveal_card_action_proba.keys()):
            self.reveal_card_action_proba[environment] = [
                1 / (COLUMN_LENGTH * len(self.card_board[0]))
                for _ in range(0, COLUMN_LENGTH * len(self.card_board[0]))
            ]
        card_position = random.choices(
            population, self.reveal_card_action_proba[environment]
        )
        return card_position[0]

    def init_game(self) -> None:
        # Player reveals two cards from its board at the beginning of the game
        environment1 = self.get_environment(None, None)
        first_card_revealed = self.select_card_to_reveal(environment1)
        self.move_history.append(
            {
                "type": "first_card_reveal_move",
                "environment": environment1,
                "actions": ["reveal_card", first_card_revealed],
            }
        )
        self.reveal_card(first_card_revealed[0], first_card_revealed[1])
        # Revealing the second card
        environment2 = self.get_environment(None, None)
        second_card_revealed = self.select_card_to_reveal(environment2)
        self.move_history.append(
            {
                "type": "second_card_reveal_move",
                "environment": environment2,
                "actions": ["reveal_card", second_card_revealed],
            }
        )
        self.reveal_card(second_card_revealed[0], second_card_revealed[1])
        # pprint(self.move_history)
        return

    def reset_board(self, card_deck: CardDeck) -> None:
        for row in self.card_board:
            for card in row:
                card_deck.discard_card(card)
        self.last_turn = False

    def play_turn(self, card_deck: CardDeck) -> None:
        # Player either draw from draw pile or take card from discard pile
        environement = self.get_environment(None, card_deck)
        draw_action = self.select_draw_action(environement)
        if draw_action == "from_discard":
            # If the card is drawn from the discard pile, Player has to exchange the card with one on the board.
            environement = self.get_environment(card_deck.discard[-1], card_deck)
            card_to_replace = self.select_card_to_replace(environement)
            self.move_history.append(
                {
                    "type": "replace_from_discard_move",
                    "environment": environement,
                    "actions": [draw_action, "replace_card", card_to_replace],
                }
            )
            drawn_card = card_deck.get_discarded_card()
            replaced_cards = self.replace_card(
                drawn_card, card_to_replace[0], card_to_replace[1]
            )
        elif draw_action == "from_deck":
            # If the card is drawn from the deck pile, Player can exchange the card with one on the board or discard the card and reveal a card on the board.
            drawn_card = card_deck.draw_card(is_visible=True)
            environement = self.get_environment(drawn_card, card_deck)
            replace_or_reveal_action = self.select_replace_or_reveal_action(
                environement
            )
            if replace_or_reveal_action == "replace_card":
                card_to_replace = self.select_card_to_replace(environement)
                self.move_history.append(
                    {
                        "type": "replace_from_deck_move",
                        "environment": environement,
                        "actions": [
                            draw_action,
                            replace_or_reveal_action,
                            card_to_replace,
                        ],
                    }
                )
                replaced_cards = self.replace_card(
                    drawn_card, card_to_replace[0], card_to_replace[1]
                )

            elif replace_or_reveal_action == "reveal_card":
                card_to_reveal = self.select_card_to_reveal(environement)
                self.move_history.append(
                    {
                        "type": "reveal_from_deck_move",
                        "environment": environement,
                        "actions": [
                            draw_action,
                            replace_or_reveal_action,
                            card_to_reveal,
                        ],
                    }
                )
                card_deck.discard_card(drawn_card)
                replaced_cards = self.reveal_card(card_to_reveal[0], card_to_reveal[1])
        for replaced_card in replaced_cards:
            card_deck.discard_card(replaced_card)
        """
        pprint("draw card proba")
        pprint(self.draw_action_proba)
        pprint("replace card proba")
        pprint(self.replace_card_action_proba)
        pprint("drop card proba")
        pprint(self.replace_or_reveal_action_proba)
        pprint("reveal card proba")
        pprint(self.reveal_card_action_proba)
        """
        return

    def update_rewards(self) -> None:
        return


def play_game(card_deck: CardDeck, player1: Player, player2: Player, i: int) -> None:
    print("Starting game", i)
    card_deck.reset_deck()
    player_list = [player1, player2]
    for player in player_list:
        player.draw_board(card_deck)
        player.init_game()
        player.show_board()
    card_deck.init_round()
    i = 1
    # TODO: better ordering function
    player_ordered_turn = (
        [player1, player2]
        if player1.compute_current_score() > player2.compute_current_score()
        else [player2, player1]
    )
    round_over = False
    while not round_over:
        print(f"__ Turn {i} ___")
        player.show_board()
        for player in player_ordered_turn:
            if player.last_turn:
                round_over = True
                break
            player.play_turn(card_deck)
            player.show_board()
            card_deck.show_deck()
            player.last_turn = not player.has_hidden_cards()
            if player.last_turn:
                print(
                    f"{player.player_name} has revealed all his board ! This is the last turn !"
                )
        i += 1
    print("Round over !")
    min_score = 1000
    best_player = None
    for player in player_ordered_turn:
        player_final_score = player.compute_final_score()
        if player_final_score < min_score:
            min_score = player_final_score
            best_player = player
            player.has_won = True
        player.reset_board(card_deck)
    print(
        f"The winner of the round is {best_player.player_name} with a score of {min_score} !"
    )
    # pprint(player1.move_history)
    # pprint(len(player2.move_history))


if __name__ == "__main__":
    card_deck = CardDeck()
    player1 = Player("test_bot_1")
    player2 = Player("test_bot_2")
    for i in range(150):
        play_game(card_deck, player1, player2, i)
