#!/usr/bin/python
from typing import List, Dict
import random
import numpy as np
import os
import re
import json
import string
import boto3
import sys

from datetime import datetime

ACTIONS = ['F', 'A']    # Fold vs All-in
PLAYERS = 4             # Numbers of players
STACK = 16              # Stack size
BB = 2                  # Big blind
SB = 1                  # Small blind
RAKE = 0.3

all_payouts = {}

class Deck:
    def __init__(self, players):
        self.players = players
        self.values = ['A', 'K', 'Q', 'J', 'T', '9', '8', '7', '6', '5', '4', '3', '2']

        self.combos = []
        for i in range(0, len(self.values)):
            for j in range(i, len(self.values)):
                combo = self.values[i] + self.values[j]
                if i != j:
                    self.combos.append(combo + 's')
                    self.combos.append(combo + 'o')
                    continue
                self.combos.append(combo)
        
        self.suits = ['s', 'd', 'h', 'c']
        self.cards = [i + j for j in self.suits for i in self.values]

    def deal(self):
        cards = random.sample(self.cards, self.players * 2)

        holdings = []

        for _ in range(0, self.players):
            card = (cards[_*2], cards[_*2+1])
            combo = ""
            if card[0][0] == card[1][0]:
                combo = card[0][0]+card[1][0]
                if combo not in self.combos:
                    combo = card[1][0]+card[0][0]
            elif card[0][1] == card[1][1]:
                combo = card[0][0]+card[1][0]+'s'
                if combo not in self.combos:
                    combo = card[1][0]+card[0][0]+'s'
            else:
                combo = card[0][0]+card[1][0]+'o'
                if combo not in self.combos:
                    combo = card[1][0]+card[0][0]+'o'
            holdings.append((combo, card))
        

        return holdings


class AOFPoker:
    @staticmethod
    def payouts(history, holdings):
        if history.startswith("FFF"):
            return [-RAKE, -RAKE, -SB-RAKE, SB-RAKE]
            
        pot = [0, 0, SB, BB]
        payouts = [-RAKE, -RAKE, -SB-RAKE, -BB-RAKE]
        place = 0
        input = ""

        for _ in range(0, len(history)):
            if history[_] == 'A':
                pot[_] = STACK
                place = _
                input += f" {holdings[_][1][0]} {holdings[_][1][1]}"

        if history.count("A") < 2:
            payouts[place] += sum(pot)
        else:
            total_in_pot = sum(pot)
            output_stream = os.popen("../poker-eval/examples/pokenum -h -t " + input)
            odds = output_stream.readline().strip().split(" ")
            odd_start = 2
            for i in range(0, len(history)):
                if history[i] == 'A':
                    payouts[i] = round(((total_in_pot)*float(odds[odd_start])-STACK)-(STACK*(1 - float(odds[odd_start]))), 4) - RAKE
                    odd_start += 1
            hhhh = []
            
            for _ in holdings:
                hhhh.append(_[1][0])
                hhhh.append(_[1][1])

            rh = '-'.join(hhhh)

            if rh not in all_payouts:
                all_payouts[rh] = {}
            all_payouts[rh][input.strip().replace(" ", "-")] = payouts
        return payouts

    @staticmethod
    def is_terminal(history: str, players: int) -> bool:
        if history.startswith("FFF"): return True
        return len(history) == players


class InformationSet():
    def __init__(self):
        self.cumulative_regrets = np.zeros(shape=len(ACTIONS))
        self.strategy_sum = np.zeros(shape=len(ACTIONS))
        self.num_actions = len(ACTIONS)

    def normalize(self, strategy: np.array) -> np.array:
        if sum(strategy) > 0:
            strategy /= sum(strategy)
        else:
            strategy = np.array([1.0 / self.num_actions] * self.num_actions)
        return strategy

    def get_strategy(self, reach_probability: float) -> np.array:
        strategy = np.maximum(0, self.cumulative_regrets)
        strategy = self.normalize(strategy)

        self.strategy_sum += reach_probability * strategy
        return strategy

    def get_average_strategy(self) -> np.array:
        return self.normalize(self.strategy_sum.copy())

    def get_average_strategy_with_threshold(self, threshold: float) -> np.array:
        avg_strat = self.get_average_strategy()
        avg_strat[avg_strat < threshold] = 0
        return self.normalize(avg_strat)

class AOFTrainer():
    def __init__(self, num_players: int):
        self.infoset_map: Dict[str, InformationSet] = {}
        self.num_players = num_players

    def reset(self):
        """reset strategy sums"""
        for n in self.infoset_map.values():
            n.strategy_sum = np.zeros(n.num_actions)

    def get_information_set(self, card_and_history: str) -> InformationSet:
        """add if needed and return"""
        if card_and_history not in self.infoset_map:
            self.infoset_map[card_and_history] = InformationSet()
        return self.infoset_map[card_and_history]

    def get_counterfactual_reach_probability(self, probs: np.array, player: int):
        """compute counterfactual reach probability"""
        return np.prod(probs[:player]) * np.prod(probs[player + 1:])

    def cfr(self, cards: List[str], history: str, reach_probabilities: np.array, active_player: int) -> np.array:
        if AOFPoker.is_terminal(history, self.num_players):
            return AOFPoker.payouts(history, cards)

        my_card = cards[active_player]


        info_set = self.get_information_set(str(my_card[0]) + history)

        strategy = info_set.get_strategy(reach_probabilities[active_player])
        next_player = (active_player + 1) % self.num_players

        counterfactual_values = [None] * len(ACTIONS)

        for ix, action in enumerate(ACTIONS):
            action_probability = strategy[ix]

            # compute new reach probabilities after this action
            new_reach_probabilities = reach_probabilities.copy()
            new_reach_probabilities[active_player] *= action_probability

            # recursively call cfr method, next player to act is the opponent
            counterfactual_values[ix] = self.cfr(cards, history + action, new_reach_probabilities, next_player)

        # Value of the current game state is just counterfactual values weighted by action probabilities
        node_values = strategy.dot(counterfactual_values)  # counterfactual_values.dot(strategy)
        for ix, action in enumerate(ACTIONS):
            cf_reach_prob = self.get_counterfactual_reach_probability(reach_probabilities, active_player)
            regrets = counterfactual_values[ix][active_player] - node_values[active_player]
            info_set.cumulative_regrets[ix] += cf_reach_prob * regrets
        return node_values

    def train(self, num_iterations: int) -> int:
        utils = np.zeros(self.num_players)
        deck = Deck(4)
        for _ in range(num_iterations):
            cards = deck.deal()
            history = ''
            reach_probabilities = np.ones(self.num_players)
            utils += self.cfr(cards, history, reach_probabilities, 0)
        return utils

def pretty_print_infoset_name(name) -> str:
    return name

if __name__ == "__main__":
    num_iterations = int(sys.argv[3])
    count = 0

    np.set_printoptions(precision=4, floatmode='fixed', suppress=True)
    np.random.seed(43)

    cfr_trainer = AOFTrainer(PLAYERS)
    deck = Deck(PLAYERS)


    while True:
        utils = cfr_trainer.train(num_iterations)

        letters = string.ascii_lowercase
        s = ''.join(random.choice(letters) for i in range(5))
        date = datetime.now().strftime("%d-%m-%Y-%H-%M")
        print("Writing results to file")
        file_name = f"payouts-{num_iterations}-{date}-{s}.json"
        with open("payouts.json", "w") as outfile:
            json.dump(all_payouts, outfile)
        all_payouts = {}

        session = boto3.Session(
            aws_access_key_id=sys.argv[1],
            aws_secret_access_key=sys.argv[2],
        )
        s3 = session.resource('s3')
        s3.meta.client.upload_file(Filename="payouts.json", Bucket='aof-payouts', Key=file_name)
