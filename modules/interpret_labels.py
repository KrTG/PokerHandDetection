from collections import namedtuple
from itertools import permutations, combinations

def get_combination(labels):
    Card = namedtuple('Card', ['color', 'rank'])

    cards = [Card(label // 13, label % 13) for label in labels]        
    permutations_of_5 = permutations(cards, r=5)
    combinations_of_4 = combinations(cards, r=4)
    combinations_of_3 = combinations(cards, r=3)
    combinations_of_2 = combinations(cards, r=2)

    flush = all(card.color == cards[0].color for card in cards)
    straight = any(
        all(p[i+1].rank == p[i].rank + 1 for i in range(4))
        for p in permutations_of_5        
    )
    four_of_a_kind = [
        c for c in combinations_of_4
        if all(card.rank == c[0].rank for card in c)        
    ]
    three_of_a_kind = [
        c for c in combinations_of_3
        if all(card.rank == c[0].rank for card in c)        
    ]

    pairs = [
        c for c in combinations_of_2
        if c[0].rank == c[1].rank        
    ]
    # 3s cant be part of 4s, 2s cant be part of 3s
    if four_of_a_kind:
        three_of_a_kind = []
        pairs = []
    if three_of_a_kind:
        pairs = [p for p in pairs if p[0].rank != three_of_a_kind[0][0].rank]    
            
    if straight and flush:
        high_card = max(card.rank for card in cards)
        if high_card == 12: # Ace 
            return "Royal flush"

        high_card_desc = get_rank(high_card)
        return "Straight flush, {} high".format(high_card_desc)
    
    if four_of_a_kind:
        high_card = max(card.rank for card in four_of_a_kind[0])
        high_card_desc = get_rank(high_card)
        return "Four of a kind, {} high".format(high_card_desc)

    if three_of_a_kind and pairs:
        high_card = max(card.rank for card in three_of_a_kind[0])
        high_card_desc = get_rank(high_card)
        return "Full house, {} high".format(high_card_desc)

    if flush:
        high_card = max(card.rank for card in cards)
        high_card_desc = get_rank(high_card)
        return "Flush, {} high".format(high_card_desc)

    if straight:
        high_card = max(card.rank for card in cards)
        high_card_desc = get_rank(high_card)
        return "Straight, {} high".format(high_card_desc)

    if three_of_a_kind:
        high_card = max(card.rank for card in three_of_a_kind[0])
        high_card_desc = get_rank(high_card)
        return "Three of a kind, {} high".format(high_card_desc)

    if len(pairs) == 2:
        first = pairs[0][0].rank
        second = pairs[1][0].rank
        if second > first:
            first, second = second, first
        
        first = get_rank(first)
        second = get_rank(second)

        return "Two pairs: {}s and {}s".format(first, second)

    if len(pairs) == 1:
        rank = get_rank(pairs[0][0].rank)

        return "Pair of {}s".format(rank)

    high_card = max(card.rank for card in cards)
    high_card_desc = get_rank(high_card)

    return "High card {}".format(high_card_desc)

def get_label(card_color, card_rank):
    color_label = {
        'kier': 0,
        'karo': 1,
        'pik': 2,
        'trefl': 3,
    }[card_color]

    rank_label = {
        '2': 0,
        '3': 1,
        '4': 2,
        '5': 3,
        '6': 4,
        '7': 5,
        '8': 6,
        '9': 7,
        '10': 8,
        'B': 9,
        'D': 10,
        'K': 11,
        'A': 12,
    }[card_rank]    

    return color_label * 13 + rank_label

def get_rank(rank_label):
    return [
        '2',
        '3',
        '4',
        '5',
        '6',
        '7',
        '8',
        '9',
        '10',
        'B',
        'D',
        'K',
        'A',
    ][rank_label]    

def get_color(color_label):
    return [
        'kier',
        'karo',
        'pik',
        'trefl',
    ][color_label]


def get_classes(label):
    return get_color(label // 13), get_rank(label % 13)        

def get_description(label):
    color, rank = get_classes(label)
    if color == None:
        return 'unknown'    
    return "{}, {}".format(color, rank)