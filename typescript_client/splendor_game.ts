// engine.ts
// Core Splendor game logic in TypeScript

import { tier1CardData, tier2CardData, tier3CardData, noblesData } from './game_data';

export const COLOR_NAMES = ['black', 'white', 'red', 'blue', 'green', 'gold'] as const;
export type Color = typeof COLOR_NAMES[number];

export enum ActionType {
  BUY_BOARD,
  BUY_RESERVE,
  TAKE_DIFF,
  TAKE_SAME,
  RESERVE_CARD,
  RESERVE_DECK,
  DISCARD,
}

export type Action = [ActionType, number | number[]];

export class Card {
  constructor(
    public level: number,
    public cost: number[],
    public points: number,
    public bonus: number
  ) {}
}

export class Noble {
  public points = 3;
  constructor(public requirement: number[]) {}
}

export class Player {
  public gems: number[] = Array(6).fill(0);
  public bonuses: number[] = Array(5).fill(0);
  public reserved: Card[] = [];
  public VPs = 0;

  constructor() {}
}

export class SplendorGame {
  static MAX_GEMS = 10;

  public boardGems: number[];
  private decks: Card[][] = [[], [], []];
  public board: Card[][] = [[], [], []];
  public nobles: Noble[] = [];
  public players: Player[];
  public currentPlayer = 0;

  constructor(numPlayers = 2) {
    const base = numPlayers === 2 ? 4 : numPlayers === 3 ? 5 : 7;
    this.boardGems = Array(5).fill(base).concat(5);
    this.players = Array.from({ length: numPlayers }, () => new Player());

    this.setupDecks();
    this.setupNobles();
    this.setupBoard();
  }

  private setupDecks(): void {
    [tier1CardData, tier2CardData, tier3CardData].forEach((data, lvl) => {
      const cards = data.map(row => new Card(lvl, row.slice(0, 5), row[5], row[6]));
      this.decks[lvl] = this.shuffle(cards);
    });
  }

  private setupNobles(): void {
    const nobles = noblesData.map(req => new Noble(req));
    this.nobles = this.shuffle(nobles).slice(0, this.players.length + 1);
  }

  private setupBoard(): void {
    for (let lvl = 0; lvl < 3; lvl++) {
      this.board[lvl] = this.decks[lvl].splice(0, Math.min(4, this.decks[lvl].length));
    }
  }

  public legalActions(pIdx: number): Action[] {
    const player = this.players[pIdx];
    const total = player.gems.reduce((s, g) => s + g, 0);
    if (total > SplendorGame.MAX_GEMS) {
      return player.gems
        .map<Action | null>((g, c) =>
          g > 0 ? [ActionType.DISCARD, c] : null
        )
        .filter((a): a is Action => a !== null);
    }

    const actions: Action[] = [];
    // TAKE_DIFF up to 3 distinct
    const available = this.boardGems
      .map((g, c) => g > 0 ? c : -1)
      .filter(c => c >= 0);
    const k = Math.min(3, available.length);
    this.combinations(available, k).forEach(combo =>
      actions.push([ActionType.TAKE_DIFF, combo])
    );
    // TAKE_SAME
    available.forEach(c => {
      if (this.boardGems[c] >= 4) actions.push([ActionType.TAKE_SAME, c]);
    });

    // RESERVE_CARD and RESERVE_DECK
    if (player.reserved.length < 3) {
      this.board.forEach((lvlBoard, lvl) => {
        lvlBoard.forEach((_, idx) =>
          actions.push([ActionType.RESERVE_CARD, [lvl, idx]])
        );
        if (this.decks[lvl].length > 0) {
          actions.push([ActionType.RESERVE_DECK, lvl]);
        }
      });
    }

    // BUY_BOARD
    this.board.forEach((lvlBoard, lvl) => {
      lvlBoard.forEach((card, idx) => {
        if (this.canAfford(player, card)) {
          actions.push([ActionType.BUY_BOARD, [lvl, idx]]);
        }
      });
    });

    // BUY_RESERVE
    player.reserved.forEach((card, idx) => {
      if (this.canAfford(player, card)) {
        actions.push([ActionType.BUY_RESERVE, idx]);
      }
    });

    return actions;
  }

  public step(action: Action): void {
    const [type, param] = action;
    const player = this.players[this.currentPlayer];

    switch (type) {
      case ActionType.TAKE_DIFF:
        this.doTake(player, param as number[]);
        break;
      case ActionType.TAKE_SAME:
        this.doTake(player, [param as number, param as number]);
        break;
      case ActionType.RESERVE_CARD: {
        const [lvl, idx] = param as number[];
        const card = this.board[lvl].splice(idx, 1)[0];
        player.reserved.push(card);
        if (this.boardGems[5] > 0) { this.boardGems[5]--; player.gems[5]++; }
        break;
      }
      case ActionType.RESERVE_DECK: {
        const lvl = param as number;
        const card = this.decks[lvl].pop()!;
        player.reserved.push(card);
        if (this.boardGems[5] > 0) { this.boardGems[5]--; player.gems[5]++; }
        break;
      }
      case ActionType.BUY_BOARD: {
        const [lvl, idx] = param as number[];
        const card = this.board[lvl].splice(idx, 1)[0];
        this.doBuy(player, card);
        break;
      }
      case ActionType.BUY_RESERVE: {
        const idx = param as number;
        const card = player.reserved.splice(idx, 1)[0];
        this.doBuy(player, card);
        break;
      }
      case ActionType.DISCARD:
        player.gems[param as number]--;
        this.boardGems[param as number]++;
        break;
    }

    this.handleNobles(player);
    this.refillBoard();
    this.currentPlayer = (this.currentPlayer + 1) % this.players.length;
  }

  private canAfford(player: Player, card: Card): boolean {
    const cost = card.cost.map((c, i) => Math.max(0, c - player.bonuses[i]));
    const remaining = cost.map((c, i) => c - player.gems[i]);
    const goldNeeded = remaining
      .map(r => Math.max(0, r))
      .reduce((s, r) => s + r, 0);
    return goldNeeded <= player.gems[5];
  }

  private doTake(player: Player, colors: number[]): void {
    colors.forEach(c => {
      this.boardGems[c]--;
      player.gems[c]++;
    });
  }

  private doBuy(player: Player, card: Card): void {
    let cost = card.cost.map((c, i) => Math.max(0, c - player.bonuses[i]));
    cost.forEach((amt, i) => {
      const pay = Math.min(amt, player.gems[i]);
      player.gems[i] -= pay;
      this.boardGems[i] += pay;
      cost[i] -= pay;
    });
    const goldNeeded = cost.reduce((s, c) => s + c, 0);
    player.gems[5] -= goldNeeded;
    this.boardGems[5] += goldNeeded;
    player.bonuses[card.bonus]++;
    player.VPs += card.points;
  }

  private handleNobles(player: Player): void {
    this.nobles = this.nobles.filter(n => {
      if (n.requirement.every((req, i) => player.bonuses[i] >= req)) {
        player.VPs += n.points;
        return false;
      }
      return true;
    });
  }

  private refillBoard(): void {
    for (let lvl = 0; lvl < 3; lvl++) {
      while (this.board[lvl].length < 4 && this.decks[lvl].length) {
        this.board[lvl].push(this.decks[lvl].pop()!);
      }
    }
  }

  private shuffle<T>(arr: T[]): T[] {
    const a = arr.slice();
    for (let i = a.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [a[i], a[j]] = [a[j], a[i]];
    }
    return a;
  }

  private combinations<T>(arr: T[], k: number): T[][] {
    const results: T[][] = [];
    const combo: T[] = [];
    function backtrack(start: number) {
      if (combo.length === k) { results.push(combo.slice()); return; }
      for (let i = start; i < arr.length; i++) {
        combo.push(arr[i]); backtrack(i + 1); combo.pop();
      }
    }
    backtrack(0);
    return results;
  }
}
