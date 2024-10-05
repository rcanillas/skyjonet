from typing import List, Dict
import random
import copy

# random.seed(42)


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

    def draw_board(self, card_deck: CardDeck) -> None:
        for _ in range(0, 3):
            col = []
            for _ in range(0, 4):
                col.append(card_deck.draw_card())
            self.card_board.append(col)

    def replace_card(self, card: Card, line: int, col: int) -> Card:
        card_to_discard = copy.copy(self.card_board[line][col])
        self.card_board[line][col] = card
        print(
            f"Player {self.player_name} replaces the card {card_to_discard.value} that was {'visible' if card_to_discard.is_visible else 'hidden'} at [{line},{col}] by a card {card.value}"
        )
        return card_to_discard

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

    def compute_final_score(self) -> int:
        score = 0
        for col in self.card_board:
            for card in col:
                score += card.value
        print(f"Player {self.player_name} final score is {score}.")
        return score

    def reveal_card(self, line, col) -> bool:
        if not self.card_board[line][col].is_visible:
            self.card_board[line][col].is_visible = True
            print(
                f"Player {self.player_name} reveals the card at [{line},{col}]: it's a {self.card_board[line][col].value} !"
            )
            return True
        else:
            return False

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


class SkyjoGame:
    def __init__(self) -> None:
        self.players: List[Player] = []
        self.players_score: List[Dict[str, int]] = []


if __name__ == "__main__":
    print("Starting game")
    card_deck = CardDeck()
    card_deck.reset_deck()
    player1 = Player("test_bot_1")
    player2 = Player("test_bot_2")
    player_list = [player1, player2]
    for player in player_list:
        player.draw_board(card_deck)
        for _ in range(2):
            # TODO: replace by better algorithm here
            while not player.reveal_card(random.randint(0, 2), random.randint(0, 3)):
                continue
        player.show_board()
        player.compute_current_score()
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
        for player in player_ordered_turn:
            if player.last_turn:
                round_over = True
                break
            if i % 3 == 1:
                print(f"{player.player_name} picking card from deck")
                card = card_deck.draw_card(is_visible=True)
                discarded_card = player.replace_card(
                    card, random.randint(0, 2), random.randint(0, 3)
                )
                card_deck.discard_card(discarded_card)
            elif i % 3 == 2:
                print(f"{player.player_name} picking card from discard pile")
                card = card_deck.get_discarded_card()
                discarded_card = player.replace_card(
                    card, random.randint(0, 2), random.randint(0, 3)
                )
                card_deck.discard_card(discarded_card)
            else:
                print(
                    f"{player.player_name} discarding picked card from deck and revealing a card from the board"
                )
                card = card_deck.draw_card(is_visible=True)
                card_deck.discard_card(card)
                while not player.reveal_card(
                    random.randint(0, 2), random.randint(0, 3)
                ):
                    continue
            player.show_board()
            player.compute_current_score()
            card_deck.show_deck()
            player.last_turn = not player.has_hidden_cards()
            if player.last_turn:
                print(f"{player.player_name} has revealed all his board ! This is the last turn !")
        i += 1
    print("Round over !")
    min_score = 1000
    best_player = None
    for player in player_ordered_turn:
        player_final_score = player.compute_final_score()
        if player_final_score < min_score:
            min_score = player_final_score
            best_player = player
    print(f"The winner of the round is {best_player.player_name} with a score of {min_score} !")
   
