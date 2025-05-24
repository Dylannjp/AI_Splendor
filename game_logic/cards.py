import numpy as np

COLOR_NAMES = ['black', 'white', 'red', 'blue', 'green']


"""
Represents a Splendor card.
Attributes:
    level: int (1,2,3)
    cost: np.ndarray shape (5,) (black, white, red, blue, green)
    VPs: int (0-5)
    bonus: int (0-4)  # 0 = black, 1 = white, 2 = red, 3 = blue, 4 = green
"""
class Card:
    def __init__(self, level, cost, VPs, bonus):
        self.level = level
        self.cost = np.array(cost, dtype=int)
        self.VPs = VPs
        self.bonus = bonus

    def __repr__(self):
        bonus_name = COLOR_NAMES[self.bonus]
        return (f"Card(level={self.level}, cost={self.cost.tolist()}, "
                f"VPs={self.VPs}, bonus='{bonus_name}')")
    
class Noble:
    def __init__(self, requirement):
        self.requirement = np.array(requirement, dtype=int)
        self.VPs = 3
    def __repr__(self):
        return (f"Noble(requirement={self.requirement.tolist()}, "
                f"VPs={self.VPs})")