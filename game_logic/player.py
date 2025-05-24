import numpy as np

"""
Represents a Splendor player.
Attributes:
    gems: np.ndarray shape (6,) (black, white, red, blue, green, gold)
    bonuses: np.ndarray shape (5,) (black, white, red, blue, green)
    points: int (0-5)
    bonus: int (0-4)  # 0 = black, 1 = white, 2 = red, 3 = blue, 4 = green
    reserved: card objects
"""
class Player:
    def __init__(self):
        self.gems=np.array([0,0,0,0,0,0])
        self.bonuses=np.array([0,0,0,0,0])
        self.VPs=0
        self.reserved=[]

    def __repr__(self):
        
        return (f"VPs(VP={self.VPs}, Gems={self.gems.tolist()}, Bonuses={self.bonuses.tolist()}, Reserved={self.reserved}")(self.VPs, self.gems.tolist(), self.gems.tolist(), self.reserved)
    
