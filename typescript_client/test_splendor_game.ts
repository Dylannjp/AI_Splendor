// test_engine.ts
import { SplendorGame, ActionType, COLOR_NAMES } from './splendor_game';

// Simple assert helper
function assert(cond: boolean, msg: string) {
  if (!cond) throw new Error('Assertion failed: ' + msg);
}

console.log('☑️  Starting manual smoke-tests…');

const game = new SplendorGame(2);

// 1) Initial board sizes
game.board.forEach((tier, lvl) => {
  console.log(` Tier ${lvl+1} cards:`, tier.length);
  assert(tier.length === 4, `Expected 4 cards at tier ${lvl+1}`);
});

// 2) Noble count
console.log(' Nobles dealt:', game.nobles.length);
assert(game.nobles.length === 3, 'Should deal players+1 nobles');

// 3) TAKE_DIFF test
const p0 = 0;
const takeDiff = game.legalActions(p0).find(a => a[0] === ActionType.TAKE_DIFF);
assert(!!takeDiff, 'Expected a TAKE_DIFF action');
{
  const beforeBoard = [...game.boardGems];
  const beforePlayer = [...game.players[p0].gems];
  game.step(takeDiff!);
  const colors = takeDiff![1] as number[];
  colors.forEach(c => {
    assert(game.boardGems[c] === beforeBoard[c] - 1,
           `Board[${c}] should drop by 1`);
    assert(game.players[p0].gems[c] === beforePlayer[c] + 1,
           `Player gems[${c}] should rise by 1`);
  });
  console.log(' ☑️  TAKE_DIFF works');
}

// 4) RESERVE_DECK test
const reserveDeck = game.legalActions(p0).find(a => a[0] === ActionType.RESERVE_DECK);
assert(!!reserveDeck, 'Expected a RESERVE_DECK action');
{
  const beforeGold = game.players[p0].gems[5];
  const beforeReserved = game.players[p0].reserved.length;
  game.step(reserveDeck!);
  assert(game.players[p0].reserved.length === beforeReserved + 1,
         'Reserved count should increment');
  assert(game.players[p0].gems[5] === beforeGold + 1,
         'Player should receive one gold token');
  console.log(' ☑️  RESERVE_DECK works');
}

console.log('🎉  All manual smoke-tests passed!');
