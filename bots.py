import random
import numpy as np
from itertools import combinations
from game_logic.splendor_game import SplendorGame, ActionType

def noble_rush_bot(game: "SplendorGame", p_idx: int):
    """
    1) If buying any face-up or reserved card right now immediately grants enough bonuses to claim
       a visible noble, do it.
    2) Otherwise, pick the single “closest” noble (smallest sum of missing bonuses). Let `target_noble`.
    3) Find *any* card (tier 0,1,2 Board or in your reserve) whose bonus is one of target_noble’s missing colors.
       • If you can BUY that card now → do it.
       • Else if you can RESERVE that card now → do it.
    4) Otherwise, take gems toward affording the cheapest‐in‐gems card (board or reserve) that yields a needed bonus.
       (Compute missing cost for that bonus‐card, then choose a “Take 3” or “Take 2/1” action toward its colors.)
    5) If all else fails, just TAKE_DIFF random valid combo.
    """
    player = game.players[p_idx]
    legal = set(game.legal_actions(p_idx))
    visible_nobles = [n for n in game.nobles if n is not None]

    # --- 1) If any immediate BUY (board/reserve) grants a noble → do it. ---
    for act in legal:
        if act[0] in (ActionType.BUY_BOARD, ActionType.BUY_RESERVE):
            if act[0] is ActionType.BUY_BOARD:
                lvl, slot = act[1]
                card = game.board_cards[lvl][slot]
            else:
                slot = act[1]
                card = player.reserved[slot]
            # Simulate the new bonus vector if we purchase that card:
            new_bonus = player.bonuses.copy()
            new_bonus[card.bonus] += 1
            for noble in visible_nobles:
                if np.all(new_bonus >= noble.requirement):
                    return act

    # --- 2) Find the “closest” noble by missing‐bonus sum. ---
    if not visible_nobles:
        # no nobles left—just fallback to a small Tier‐0 pickup or random
        return random.choice(list(legal))

    missing_list = []
    for noble in visible_nobles:
        diff = noble.requirement - player.bonuses
        diff = np.clip(diff, 0, None)
        missing_list.append((diff.sum(), noble))
    missing_list.sort(key=lambda x: x[0])
    closest_missing, target_noble = missing_list[0]
    needed_colors = [i for i in range(5) if (target_noble.requirement[i] - player.bonuses[i]) > 0]

    # --- 3) Find any card (board or in reserve) whose bonus is in needed_colors. ---
    #    Then either buy or reserve it.
    #   a) Look face-up boards (tier 0,1,2):
    for lvl in (0, 1, 2):
        for slot, card in enumerate(game.board_cards[lvl]):
            if not card:
                continue
            if card.bonus in needed_colors:
                buy_act = (ActionType.BUY_BOARD, (lvl, slot))
                if buy_act in legal:
                    return buy_act
                reserve_act = (ActionType.RESERVE_CARD, (lvl, slot))
                if reserve_act in legal:
                    return reserve_act

    #   b) Look in your own reserve for a card whose bonus is needed:
    for idx, card in enumerate(player.reserved):
        if card and card.bonus in needed_colors:
            buy_res_act = (ActionType.BUY_RESERVE, idx)
            if buy_res_act in legal:
                return buy_res_act

    # --- 4) Otherwise: find the *cheapest* (in gem‐cost terms) card (board or in reserve)
    #             that yields a needed bonus.  Then “take gems” toward that specific card’s missing cost.
    best_choice = None
    best_missing = 999
    # a) Check face-up boards:
    for lvl in (0, 1, 2):
        for slot, card in enumerate(game.board_cards[lvl]):
            if not card or card.bonus not in needed_colors:
                continue
            # compute how many raw gems needed to buy this card from the board right now:
            cost_vec = card.cost - player.bonuses
            cost_vec = np.clip(cost_vec, 0, None)
            # sub non-gold gems:
            rem = cost_vec[:5] - player.gems[:5]
            rem = np.clip(rem, 0, None)
            total_gold_needed = rem.sum() - player.gems[5]
            missing_gems = max(0, total_gold_needed) + rem.sum()  # approximate “effort”
            if missing_gems < best_missing:
                best_missing = missing_gems
                best_choice = ("board", lvl, slot, card)

    # b) Check your reserve (if it has a needed_bonus card):
    for idx, card in enumerate(player.reserved):
        if card and card.bonus in needed_colors:
            cost_vec = card.cost - player.bonuses
            cost_vec = np.clip(cost_vec, 0, None)
            rem = cost_vec[:5] - player.gems[:5]
            rem = np.clip(rem, 0, None)
            total_gold_needed = rem.sum() - player.gems[5]
            missing_gems = max(0, total_gold_needed) + rem.sum()
            if missing_gems < best_missing:
                best_missing = missing_gems
                best_choice = ("reserve", idx, None, card)

    # If we found some “best_choice,” take gems toward it:
    if best_choice is not None:
        _, lvl_or_idx, _, card = best_choice
        cost_vec = card.cost - player.bonuses
        cost_vec = np.clip(cost_vec, 0, None)
        # "rem_colors" is the multiset of colors we still lack:
        raw_cost = cost_vec.copy()
        raw_cost[:5] = np.clip(raw_cost[:5] - player.gems[:5], 0, None)
        # remove gold need
        gold_short = raw_cost[:5].sum() - player.gems[5]
        if gold_short < 0:
            gold_short = 0
        # Now pick up to 3 colors among those with raw_cost > 0
        missing_colors = [i for i in range(5) if raw_cost[i] > 0]
        board_gems = game.board_gems

        # Try TAKE_DIFF of 3 missing colors
        for combo in combinations(missing_colors, min(3, len(missing_colors))):
            if len(combo) == 3 and all(board_gems[c] >= 1 for c in combo):
                if (ActionType.TAKE_DIFF, combo) in legal:
                    return (ActionType.TAKE_DIFF, combo)
        # Try TAKE_DIFF of 2 or 1 if 3 isn’t possible
        for r in (2, 1):
            for combo in combinations(missing_colors, r):
                if all(board_gems[c] >= 1 for c in combo):
                    if (ActionType.TAKE_DIFF, combo) in legal:
                        return (ActionType.TAKE_DIFF, combo)

    # --- 5) Fallback: pick any TAKE_DIFF (first valid), or random discard if forced, else random legal. ---
    for act in legal:
        if act[0] == ActionType.TAKE_DIFF:
            return act
    for act in legal:
        if act[0] == ActionType.DISCARD:
            return act
    return random.choice(list(legal))

def tier_rush_bot(game: "SplendorGame", p_idx: int):
    """
    1) Score each face-up Tier 2/3 card by (VP / missing‐gems).
       Let (tgt_lvl, tgt_slot, tgt_card, tgt_score) be best.
    2) If BUY_BOARD(tgt_lvl,tgt_slot) ∈ legal → do it.
    3) Otherwise: consider the best Tier 1 board card that reduces “turns-to-afford” for tgt_card.
       • Compute `current_missing = sum(max(0, cost - (bonuses+gems)))`.
       • For each Tier 1 board card whose bonus is in tgt_card’s missing colors, simulate buying that card:
           new_bonuses = bonuses + unit_vector(card1.bonus)
           new_missing = sum(max(0, tgt_card.cost - new_bonuses - gems_after_purchase))
         Convert missing into “turns_to_get” ≈ ceil(new_missing / 3).  
       • Pick the Tier 1 action that yields minimal “turns_to_get” if that is strictly less than baseline.
       If such a Tier 1 BUY is ∈ legal → do it.
    4) Else if RESERVE tgt_card ∈ legal → do that.
    5) Else TAKE_DIFF toward tgt_card’s missing cost (same as v1).
    """
    player = game.players[p_idx]
    legal = set(game.legal_actions(p_idx))

    # 1) Gather all face-up Tier 2+3 candidates and rank by (VP / missing_gems).
    candidates = []
    for lvl in (1, 2):
        for slot, card in enumerate(game.board_cards[lvl]):
            if not card:
                continue
            cost_vec = card.cost - player.bonuses
            cost_vec = np.clip(cost_vec, 0, None)
            # subtract non-gold first
            rem = cost_vec[:5] - player.gems[:5]
            rem = np.clip(rem, 0, None)
            gold_short = rem.sum() - player.gems[5]
            missing_gems = max(0, gold_short) + rem.sum()
            if missing_gems == 0:
                score = float('inf')  # can buy immediately → top priority
            else:
                score = card.VPs / float(missing_gems)
            candidates.append((score, lvl, slot, card, missing_gems))

    if not candidates:
        return random.choice(list(legal))
    candidates.sort(key=lambda x: -x[0])
    _, tgt_lvl, tgt_slot, tgt_card, base_missing = candidates[0]
    buy_tgt = (ActionType.BUY_BOARD, (tgt_lvl, tgt_slot))
    reserve_tgt = (ActionType.RESERVE_CARD, (tgt_lvl, tgt_slot))

    # 2) If can buy it now → do it.
    if buy_tgt in legal:
        return buy_tgt

    # 3) Otherwise, check each Tier 1 board card that helps reduce “turns_to_afford” for tgt_card:
    #    baseline turns_to_afford ≈ ceil(base_missing / 3)
    from math import ceil
    baseline_turns = ceil(base_missing / 3.0)

    best_t1_action = None
    best_reduced_turns = baseline_turns
    # For each face-up Tier 1 card whose bonus belongs to tgt_card’s missing colors:
    cost_needed = tgt_card.cost - player.bonuses
    cost_needed = np.clip(cost_needed, 0, None)
    # Determine which colors are truly “limiting”:
    rem_needed = cost_needed[:5] - player.gems[:5]
    rem_needed = np.clip(rem_needed, 0, None)
    missing_colors = [i for i in range(5) if rem_needed[i] > 0]

    for slot1, card1 in enumerate(game.board_cards[0]):
        if not card1 or card1.bonus not in missing_colors:
            continue
        buy_t1 = (ActionType.BUY_BOARD, (0, slot1))
        if buy_t1 not in legal:
            continue
        # Simulate: new bonuses and new gems after purchasing card1
        new_bonus = player.bonuses.copy()
        new_bonus[card1.bonus] += 1
        # subtract player.gems used for card1:
        #   compute actual payment:
        cost1 = card1.cost - player.bonuses
        cost1 = np.clip(cost1, 0, None)
        pay_non_gold = np.minimum(cost1[:5], player.gems[:5])
        leftover_cost = cost1[:5] - pay_non_gold
        gold_used = leftover_cost.sum()
        # Build new gem array:
        new_gems = player.gems.copy()
        new_gems[:5] -= pay_non_gold
        new_gems[5] -= gold_used

        # compute missing_gems’ for tgt_card under new_bonus & new_gems:
        new_cost = tgt_card.cost - new_bonus
        new_cost = np.clip(new_cost, 0, None)
        rem2 = new_cost[:5] - new_gems[:5]
        rem2 = np.clip(rem2, 0, None)
        gold_short2 = rem2.sum() - new_gems[5]
        missing2 = max(0, gold_short2) + rem2.sum()
        new_turns = ceil(missing2 / 3.0)
        if new_turns < best_reduced_turns:
            best_reduced_turns = new_turns
            best_t1_action = buy_t1

    if best_t1_action is not None:
        return best_t1_action

    # 4) Else if can RESERVE the target Tier 2/3 → do it.
    if reserve_tgt in legal:
        return reserve_tgt

    # 5) Otherwise TAKE_DIFF toward tgt_card’s missing cost:
    new_bonus = player.bonuses.copy()
    cost_vec = tgt_card.cost - new_bonus
    cost_vec = np.clip(cost_vec, 0, None)
    rem = cost_vec[:5] - player.gems[:5]
    rem = np.clip(rem, 0, None)
    missing_colors = [i for i in range(5) if rem[i] > 0]
    board_gems = game.board_gems

    # a) TAKE_DIFF of 3 missing
    for combo in combinations(missing_colors, min(3, len(missing_colors))):
        if len(combo) == 3 and all(board_gems[c] >= 1 for c in combo):
            if (ActionType.TAKE_DIFF, combo) in legal:
                return (ActionType.TAKE_DIFF, combo)
    # b) TAKE_DIFF of 2 or 1
    for r in (2, 1):
        for combo in combinations(missing_colors, r):
            if all(board_gems[c] >= 1 for c in combo):
                if (ActionType.TAKE_DIFF, combo) in legal:
                  return (ActionType.TAKE_DIFF, combo)

    # 6) fallback: DISCARD if forced, else random
    for act in legal:
        if act[0] == ActionType.DISCARD:
            return act
    return random.choice(list(legal))


def noble_blocker_bot(game: "SplendorGame", p_idx: int):
    """
    1) Check if opponent is exactly 1 bonus away from any visible noble.
       If so, find a board or deck card that grants that bonus and reserve/buy it.
    2) Otherwise, fall back to noble_rush_bot (so our own noble‐rush logic).
    """
    opponent = 1 - p_idx
    opp = game.players[opponent]
    legal_us = set(game.legal_actions(p_idx))

    visible_nobles = [n for n in game.nobles if n is not None]
    # 1) For each noble, see if opponent needs exactly 1 more bonus to claim:
    for noble in visible_nobles:
        diff = noble.requirement - opp.bonuses
        diff = np.clip(diff, 0, None)
        if diff.sum() == 1:
            # Find which color they need
            color_needed = int(np.where(diff > 0)[0][0])
            # Find any face-up card (tier 0–2) whose bonus == color_needed
            for lvl in (0, 1, 2):
                for slot, card in enumerate(game.board_cards[lvl]):
                    if not card:
                        continue
                    if card.bonus == color_needed:
                        buy_act = (ActionType.BUY_BOARD, (lvl, slot))
                        if buy_act in legal_us:
                            return buy_act
                        reserve_act = (ActionType.RESERVE_CARD, (lvl, slot))
                        if reserve_act in legal_us:
                            return reserve_act
            # If none on face-up board, they might get it from a deck “reserve deck”:
            for lvl in (0, 1, 2):
                # “reserve deck” implicitly grants a random card of that tier; if that card’s bonus == color_needed,
                # we cannot see it in advance—so skip. Only board cards are blockable deterministically.
                # So do nothing else here.
                pass

    # 2) Else fallback to your own noble‐rush logic:
    return noble_rush_bot(game, p_idx)

def gem_blocker_bot(game: "SplendorGame", p_idx: int):
    """
    1) Identify opponent’s most‐likely next face-up card (tier 0–2) using the same “missing_cost" metric.
    2) Compute which raw colors the opponent still lacks. 
       If any of those colors are available on the board and we can TAKE_DIFF or TAKE_SAME them, do so.
    3) If neither “take” is possible (e.g. not enough copies of a single color for TAKE_SAME, or not enough different colors for TAKE_DIFF),
       fallback to tier_rush_bot (or any default).
    """
    opponent = 1 - p_idx
    opp = game.players[opponent]
    legal = set(game.legal_actions(p_idx))

    # 1) Identify opponent’s top face-up‐card target by missing‐gems:
    opp_candidates = []
    for lvl in (0, 1, 2):
        for slot, card in enumerate(game.board_cards[lvl]):
            if not card:
                continue
            cost_needed = card.cost - opp.bonuses
            cost_needed = np.clip(cost_needed, 0, None)
            rem = cost_needed[:5] - opp.gems[:5]
            rem = np.clip(rem, 0, None)
            gold_short = rem.sum() - opp.gems[5]
            missing_gems = max(0, gold_short) + rem.sum()
            opp_candidates.append((missing_gems, -card.VPs, lvl, slot))

    if not opp_candidates:
        return tier_rush_bot(game, p_idx)

    opp_candidates.sort(key=lambda x: (x[0], x[1]))
    miss, negvp, tgt_lvl, tgt_slot = opp_candidates[0]
    tgt_card = game.board_cards[tgt_lvl][tgt_slot]

    # 2) Compute which raw colors the opponent still lacks:
    cost_needed = tgt_card.cost - opp.bonuses
    cost_needed = np.clip(cost_needed, 0, None)
    # account for opponent’s non-gold gems:
    rem = cost_needed[:5] - opp.gems[:5]
    rem = np.clip(rem, 0, None)
    # any color i with rem[i] > 0 is a color they still need:
    missing_colors = [i for i in range(5) if rem[i] > 0]
    board_gems = game.board_gems

    # a) If there are at least three distinct missing_colors on the board, TAKE_DIFF those three:
    for combo in combinations(missing_colors, min(3, len(missing_colors))):
        if len(combo) == 3 and all(board_gems[c] >= 1 for c in combo):
            act = (ActionType.TAKE_DIFF, combo)
            if act in legal:
                return act

    # b) Otherwise, if any one missing_color has at least 4 gems on board, TAKE_SAME that color:
    for color in missing_colors:
        if board_gems[color] >= 4:
            act = (ActionType.TAKE_SAME, color)
            if act in legal:
                return act

    # c) If forced to discard, do so:
    for act in legal:
        if act[0] == ActionType.DISCARD:
            return act

    # 3) Fallback to a balanced rush:
    return tier_rush_bot(game, p_idx)

import random
import numpy as np
from itertools import combinations
from game_logic.splendor_game import ActionType

def predictive_mix_bot(game: "SplendorGame", p_idx: int, mix_ratio: float = 0.5):
    """
    1. Look at the opponent’s reserved cards + bonuses, identify which face-up card on board they are closest to buying:
         • For each face-up card (all tiers), compute “missing_cost = sum(max(0, cost - (opp.bonuses + opp.gems[0..4])) - opp.gems[5])”.
         • The card with smallest missing_cost is their likely next target.
    2. If we can RESERVE that card ∈ our legal → do so (block).
    3. Else: flip a coin (mix_ratio) → noble_rush_bot or tier_rush_bot.
    """
    opponent = 1 - p_idx
    opp = game.players[opponent]
    legal_us = set(game.legal_actions(p_idx))

    # 1) Score every face-up card (all tiers) by “how close opp is to affording it”:
    opp_candidates = []
    for lvl in (0, 1, 2):
        for slot, card in enumerate(game.board_cards[lvl]):
            if not card:
                continue
            cost_needed = card.cost - opp.bonuses
            cost_needed = np.clip(cost_needed, 0, None)
            # Subtract opponent’s “spendable” gems (non-gold first, then gold)
            non_gold = opp.gems[:5]
            rem = cost_needed - non_gold
            rem = np.clip(rem, 0, None)
            gold_short = rem.sum() - opp.gems[5]
            missing_cost = max(0, gold_short)
            # tie-break: if missing_cost is equal, pick higher VP
            opp_candidates.append((missing_cost, -card.VPs, lvl, slot))

    if opp_candidates:
        opp_candidates.sort(key=lambda x: (x[0], x[1]))
        miss, negvp, tgt_lvl, tgt_slot = opp_candidates[0]
        block_reserve = (ActionType.RESERVE_CARD, (tgt_lvl, tgt_slot))
        if block_reserve in legal_us:
            return block_reserve

    # 2) If no blocking opportunity, mix between noble or tier rush
    if random.random() < mix_ratio:
        return noble_rush_bot(game, p_idx)
    else:
        return tier_rush_bot(game, p_idx)


def mix_blocker_bot(game: "SplendorGame", p_idx: int):
    """
    1. Identify opponent’s single most urgent face-up card target (use same logic as predictive_mix_bot).
       If we can RESERVE it, do so.
    2. Identify if opponent is 1 bonus away from any noble—if so TAKE_DIFF that color if possible.
    3. Otherwise fallback to predictive_mix_bot.
    """
    opponent = 1 - p_idx
    opp = game.players[opponent]
    legal_us = set(game.legal_actions(p_idx))

    # A) “Block” opponent’s next likely face-up card (all tiers):
    opp_candidates = []
    for lvl in (0, 1, 2):
        for slot, card in enumerate(game.board_cards[lvl]):
            if not card:
                continue
            cost_needed = card.cost - opp.bonuses
            cost_needed = np.clip(cost_needed, 0, None)
            non_gold = opp.gems[:5]
            rem = cost_needed - non_gold
            rem = np.clip(rem, 0, None)
            gold_short = rem.sum() - opp.gems[5]
            missing_cost = max(0, gold_short)
            opp_candidates.append((missing_cost, -card.VPs, lvl, slot))

    if opp_candidates:
        opp_candidates.sort(key=lambda x: (x[0], x[1]))
        miss, negvp, tgt_lvl, tgt_slot = opp_candidates[0]
        block_res = (ActionType.RESERVE_CARD, (tgt_lvl, tgt_slot))
        if block_res in legal_us:
            return block_res

    # B) “Block” opponent from claiming noble:
    visible_nobles = [n for n in game.nobles if n is not None]
    for noble in visible_nobles:
        diff = noble.requirement - opp.bonuses
        diff = np.clip(diff, 0, None)
        if diff.sum() == 1:
            color_needed = int(np.where(diff > 0)[0][0])
            if game.board_gems[color_needed] >= 1:
                if (ActionType.TAKE_DIFF, (color_needed,)) in legal_us:
                    return (ActionType.TAKE_DIFF, (color_needed,))

    # C) fallback to predictive_mix_bot
    return predictive_mix_bot(game, p_idx)

def random_bot(game: "SplendorGame", p_idx: int):
    """
    Pick any legal action at random.
    """
    legal = game.legal_actions(p_idx)
    return random.choice(legal)
