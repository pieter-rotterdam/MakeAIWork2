#!/usr/bin/env python

import random
import numpy as np

class Dobbelsteen:
    
    def __init__(self):
        self.values = set(range(1, 7))  
        self.roll()
        
        # https://realpython.com/python-dice-roll
        self.faces = {

            1: (

                "┌─────────┐\n"
                "│         │\n"
                "│    ●    │\n"
                "│         │\n"
                "└─────────┘\n"
            ),

            2: (

                "┌─────────┐\n"
                "│  ●      │\n"
                "│         │\n"
                "│      ●  │\n"
                "└─────────┘"
            ),

            3: (

                "┌─────────┐\n"
                "│  ●      │\n"
                "│    ●    │\n"
                "│      ●  │\n"
                "└─────────┘"
            ),

            4: (

                "┌─────────┐\n"
                "│  ●   ●  │\n"
                "│         │\n"
                "│  ●   ●  │\n"
                "└─────────┘"

            ),

            5: (

                "┌─────────┐\n"
                "│  ●   ●  │\n"
                "│    ●    │\n"
                "│  ●   ●  │\n"
                "└─────────┘"
            ),

            6: (

                "┌─────────┐\n"
                "│  ●   ●  │\n"
                "│  ●   ●  │\n"
                "│  ●   ●  │\n"
                "└─────────┘"

            )

        }

    def getList(self):    
        return list(self.values) #hier wordt een lijst van de set met range 1-7 aangeroepen
    
    def roll(self):
        self.number = random.choice(self.getList()) # hier kiest hij een random no uit de lijst
        # self.history = 
        
    def getNumber(self):
        return self.number # hier geeft hij de waarde die in roll random bepaald is terug als een variabele
    
    def getHistory(self):
        return self.number
    
    def show(self):        
        return str(self.faces.get(self.number)) # hier matcht hij nummer met key vd dobbelsteen voor het plaatje

def main():
    d = Dobbelsteen()
    d.roll()
    d.show()

if __name__ == main():
    main()                