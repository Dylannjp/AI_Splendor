import numpy as np

# each card row: [black cost, white cost, red cost, blue cost, green cost, points, color]
# color index: 0 = black, 1 = white, 2 = red, 3 = blue, 4 = green
tier1_card_data = np.array([
    [0, 1, 1, 1, 1, 0, 0],  # card 1 - black tier1 start
    [0, 1, 1, 2, 1, 0, 0],  # card 2
    [0, 2, 1, 2, 0, 0, 0],  # card 3
    [1, 0, 3, 0, 1, 0, 0],  # card 4
    [0, 0, 1, 0, 2, 0, 0],  # card 5
    [0, 2, 0, 0, 2, 0, 0],  # card 6
    [0, 0, 0, 0, 3, 0, 0],  # card 7
    [0, 0, 0, 4, 0, 1, 0],  # card 8 - black tier1 end

    [1, 0, 1, 1, 1, 0, 1],  # card 9 - white tier1 start
    [1, 0, 1, 1, 2, 0, 1],  # card 10
    [1, 0, 0, 2, 2, 0, 1],  # card 11
    [1, 3, 0, 1, 0, 0, 1],  # card 12
    [1, 0, 2, 0, 0, 0, 1],  # card 13
    [2, 0, 0, 2, 0, 0, 1],  # card 14
    [0, 0, 0, 3, 0, 0, 1],  # card 15
    [0, 0, 0, 0, 4, 1, 1],  # card 16 - white tier1 end

    [1, 1, 0, 0, 0, 0, 2],  # card 17 - red tier1 start
    [1, 2, 0, 0, 0, 0, 2],  # card 18
    [2, 2, 0, 0, 0, 0, 2],  # card 19
    [3, 1, 1, 0, 0, 0, 2],  # card 20
    [0, 0, 0, 2, 1, 0, 2],  # card 21
    [0, 2, 2, 0, 0, 0, 2],  # card 22
    [0, 3, 0, 0, 0, 0, 2],  # card 23
    [0, 4, 0, 0, 0, 1, 2],  # card 24 - red tier1 end

    [1, 1, 1, 0, 1, 0, 3],  # card 25 - blue tier1 start
    [1, 1, 2, 0, 1, 0, 3],  # card 26
    [0, 1, 2, 0, 2, 0, 3],  # card 27
    [0, 0, 1, 1, 3, 0, 3],  # card 28
    [2, 1, 0, 0, 0, 0, 3],  # card 29
    [2, 0, 0, 0, 2, 0, 3],  # card 30
    [3, 0, 0, 0, 0, 0, 3],  # card 31
    [0, 0, 4, 0, 0, 1, 3],  # card 32 - blue tier1 end

    [1, 1, 1, 1, 0, 0, 4],  # card 33 - green tier1 start
    [2, 1, 1, 1, 0, 0, 4],  # card 34
    [2, 0, 2, 1, 0, 0, 4],  # card 35
    [0, 1, 0, 3, 0, 0, 4],  # card 36
    [0, 2, 0, 1, 0, 0, 4],  # card 37
    [0, 0, 2, 2, 0, 0, 4],  # card 38
    [0, 0, 3, 0, 0, 0, 4],  # card 39
    [4, 0, 0, 0, 0, 1, 4],  # card 40 - green tier1 end
])

tier2_card_data = np.array([
    [0, 3, 0, 2, 2, 1, 0],  # card 1 - black tier2 start
    [2, 3, 0, 0, 3, 1, 0],  # card 2
    [0, 0, 2, 1, 4, 2, 0],  # card 3
    [0, 0, 3, 0, 5, 2, 0],  # card 4
    [0, 5, 0, 0, 0, 2, 0],  # card 5
    [6, 0, 0, 0, 0, 3, 0],  # card 6 - black tier2 end

    [2, 0, 2, 0, 3, 1, 1],  # card 7 - white tier2 start
    [0, 2, 3, 3, 0, 1, 1],  # card 8
    [2, 0, 4, 0, 1, 2, 1],  # card 9
    [3, 0, 5, 0, 0, 2, 1],  # card 10
    [0, 0, 5, 0, 0, 2, 1],  # card 11
    [0, 6, 0, 0, 0, 3, 1],  # card 12 - white tier2 end

    [3, 2, 2, 0, 0, 1, 2],  # card 13 - red tier2 start
    [3, 0, 2, 3, 0, 1, 2],  # card 14
    [0, 1, 0, 4, 2, 2, 2],  # card 15
    [5, 3, 0, 0, 0, 2, 2],  # card 16
    [5, 0, 0, 0, 0, 2, 2],  # card 17
    [0, 0, 6, 0, 0, 3, 2],  # card 18 - red tier2 end

    [0, 0, 3, 2, 2, 1, 3],  # card 19 - blue tier2 start
    [3, 0, 0, 2, 3, 1, 3],  # card 20
    [0, 5, 0, 3, 0, 2, 3],  # card 21
    [4, 2, 1, 0, 0, 2, 3],  # card 22
    [0, 0, 0, 5, 0, 2, 3],  # card 23
    [0, 0, 0, 6, 0, 3, 3],  # card 24 - blue tier2 end

    [0, 3, 3, 0, 2, 1, 4],  # card 25 - green tier2 start
    [2, 2, 0, 3, 0, 1, 4],  # card 26
    [1, 4, 0, 2, 0, 2, 4],  # card 27
    [0, 0, 0, 5, 3, 2, 4],  # card 28
    [0, 0, 0, 0, 5, 2, 4],  # card 29
    [0, 0, 0, 0, 6, 3, 4],  # card 30 - green tier2 end
])

tier3_card_data = np.array([
    [0, 3, 3, 3, 5, 3, 0],  # card 1 - black tier3 start
    [0, 0, 7, 0, 0, 4, 0],  # card 2 
    [3, 0, 6, 0, 3, 4, 0],  # card 3
    [3, 0, 7, 0, 0, 5, 0],  # card 4 - black tier3 end

    [3, 0, 5, 3, 3, 3, 1],  # card 5 - white tier3 start
    [7, 0, 0, 0, 0, 4, 1],  # card 6
    [6, 3, 3, 0, 0, 4, 1],  # card 7
    [7, 3, 0, 0, 0, 5, 1],  # card 8 - white tier3 end

    [3, 3, 0, 5, 3, 3, 2],  # card 9 - red tier3 start
    [0, 0, 0, 0, 7, 4, 2],  # card 10
    [0, 0, 3, 3, 6, 4, 2],  # card 11
    [0, 0, 3, 0, 7, 5, 2],  # card 12 - red tier3 end

    [5, 3, 3, 0, 3, 3, 3],  # card 13 - blue tier3 start
    [0, 7, 0, 0, 0, 4, 3],  # card 14
    [3, 6, 0, 3, 0, 4, 3],  # card 15
    [0, 7, 0, 3, 0, 5, 3],  # card 16 - blue tier3 end

    [3, 3, 0, 5, 3, 3, 4],  # card 17 - green tier3 start
    [0, 0, 0, 0, 7, 4, 4],  # card 18
    [0, 0, 3, 3, 6, 4, 4],  # card 19
    [0, 0, 3, 0, 7, 5, 4],  # card 20 - green tier3 end
])

# each noble row = [black cost, white cost, red cost, blue cost, green cost], points are always 3 no need to encode it.
nobles_data = np.array([
    [0, 0, 4, 0, 4], # Mary Stuart
    [3, 3, 3, 0, 0], # Charles Quint - Karl V
    [0, 4, 0, 4, 0], # Macchiavelli
    [4, 4, 0, 0, 0], # Isabel of Castille
    [0, 0, 0, 4, 4], # Soliman the Magnificent
    [0, 0, 3, 3, 3], # Catherine of Medicis
    [0, 3, 0, 3, 3], # Anne of Brittany
    [4, 0, 4, 0, 0], # Henry VIII
    [3, 3, 0, 3, 0], # Elisabeth of Austria
    [3, 0, 3, 0, 3], # Francis 1 of France
])