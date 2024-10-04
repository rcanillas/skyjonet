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

    def compute_score(self) -> int:
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

    def reveal_card(self, line, col) -> bool:
        print(line, col)
        if not self.card_board[line][col].is_visible:
            self.card_board[line][col].is_visible = True
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
        print(has_hidden_cards)
        return has_hidden_cards


class SkyjoGame:
    def __init__(self) -> None:
        self.players: List[Player] = []
        self.players_score: List[Dict[str, int]] = []


if __name__ == "__main__":
    print("Starting game")
    card_deck = CardDeck()
    card_deck.reset_deck()
    player = Player("test_bot")
    player.draw_board(card_deck)
    for _ in range(2):
        # TODO: replace by better algorithm here
        while not player.reveal_card(random.randint(0, 2), random.randint(0, 3)):
            continue
    player.show_board()
    player.compute_score()
    card_deck.init_round()
    i = 1
    while player.has_hidden_cards():
        print(f"__ Turn {i} ___")
        if i % 2 == 0:
            print("picking card from deck")
            card = card_deck.draw_card(is_visible=True)
        else:
            print("picking card from discard pile")
            card = card_deck.get_discarded_card()

        discarded_card = player.replace_card(
            card, random.randint(0, 2), random.randint(0, 3)
        )
        card_deck.discard_card(discarded_card)
        player.show_board()
        player.compute_score()
        card_deck.show_deck()
        i += 1
