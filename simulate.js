'use strict';

// =============================================================================
// Pokemon Showdown Gym Leader Simulation
// 5 starter lines vs canonical gym leaders from Gen 1–5
// 1000 iterations × 5 starters × 8 gyms × 10 battles = 400,000 total battles
// =============================================================================

const { BattleStream, getPlayerStreams, PRNG, Dex } = require('./dist/sim/index');
const { RandomPlayerAI } = require('./dist/sim/tools/random-player-ai');
const fs = require('fs');
const path = require('path');

// =============================================================================
// CONFIG
// =============================================================================
const TOTAL_ITERATIONS = 1000;
const BATTLES_PER_GYM = 10;
const CONCURRENCY = 50;
const OUTPUT_DIR = path.join(__dirname, 'output');

// Pre-load gen-specific Dex objects for the damage AI
const GEN_DEX = {
  1: Dex.mod('gen1'),
  2: Dex.mod('gen2'),
  3: Dex.mod('gen3'),
  4: Dex.mod('gen4'),
  5: Dex.mod('gen5'),
};

// =============================================================================
// DAMAGE-FIRST AI FOR STARTER
// Picks the move with highest base power; falls back to random on tie
// =============================================================================
class DamageFirstAI extends RandomPlayerAI {
  constructor(playerStream, options, genNum) {
    super(playerStream, options);
    this.dex = GEN_DEX[genNum] || Dex;
  }

  chooseMove(active, moves) {
    let bestChoice = null;
    let bestPower = -1;
    for (const m of moves) {
      const moveData = this.dex.moves.get(m.move.move);
      const power = moveData ? (moveData.basePower || 0) : 0;
      if (power > bestPower) {
        bestPower = power;
        bestChoice = m;
      }
    }
    return bestChoice ? bestChoice.choice : this.prng.sample(moves).choice;
  }
}

// =============================================================================
// TEAM PACKING HELPERS
// Format: NICKNAME|SPECIES|ITEM|ABILITY|MOVES|NATURE|EVS|GENDER|IVS|SHINY|LEVEL
// =============================================================================
function packMon(species, item, ability, moves, nature, evs, ivs, level) {
  const evsStr = evs ? evs.join(',') : '';
  const ivsStr = ivs ? ivs.join(',') : '';
  // Trailing | provides the HAPPINESS,POKEBALL,... field (blank) required by Teams.unpack
  return `${species}||${item || ''}|${ability || ''}|${moves.join(',')}|${nature || ''}|${evsStr}||${ivsStr}||${level || ''}|`;
}
function packTeam(mons) { return mons.join(']'); }

// Gen 1-2: max stat experience (252 per stat), default DVs (blank = 15 DVs)
const MAX_EVS = [252, 252, 252, 252, 252, 252];

// Helpers per generation tier
function g1mon(species, moves, level) {
  return packMon(species, '', '', moves, '', MAX_EVS, null, level);
}
function g2mon(species, item, moves, level) {
  return packMon(species, item || '', '', moves, '', MAX_EVS, null, level);
}
// Gen 3-5: blank EVs (0), blank IVs (31)
function gXmon(species, item, ability, moves, nature, level) {
  return packMon(species, item || '', ability || '', moves, nature || '', null, null, level);
}

// =============================================================================
// GYM LEADER TEAMS
// Indexed: GYM_LEADERS[starterIndex (0-4)][gymIndex (0-7)]
// =============================================================================
const GYM_LEADERS = [

  // ─── GEN 1: Bulbasaur line (Red/Blue) ───────────────────────────────────────
  [
    { // 1. Brock (Pewter)
      name: 'Brock', format: 'gen1customgame', aceLevel: 14,
      team: packTeam([
        g1mon('Geodude', ['rockthrow', 'defensecurl'], 12),
        g1mon('Onix', ['tackle', 'screech', 'bind', 'bide'], 14),
      ]),
    },
    { // 2. Misty (Cerulean)
      name: 'Misty', format: 'gen1customgame', aceLevel: 21,
      team: packTeam([
        g1mon('Staryu', ['tackle', 'watergun', 'harden'], 18),
        g1mon('Starmie', ['watergun', 'bubblebeam', 'psychic', 'harden'], 21),
      ]),
    },
    { // 3. Lt. Surge (Vermilion)
      name: 'Lt. Surge', format: 'gen1customgame', aceLevel: 24,
      team: packTeam([
        g1mon('Voltorb', ['sonicboom', 'screech', 'tackle'], 21),
        g1mon('Pikachu', ['thundershock', 'quickattack', 'thunderwave', 'growl'], 18),
        g1mon('Raichu', ['thunderbolt', 'quickattack', 'thunderwave', 'bodyslam'], 24),
      ]),
    },
    { // 4. Erika (Celadon)
      name: 'Erika', format: 'gen1customgame', aceLevel: 29,
      team: packTeam([
        g1mon('Victreebel', ['wrap', 'razorleaf', 'sleeppowder', 'poisonpowder'], 29),
        g1mon('Tangela', ['constrict', 'absorb', 'growth', 'stunspore'], 24),
        g1mon('Vileplume', ['petaldance', 'sleeppowder', 'poisonpowder', 'stunspore'], 29),
      ]),
    },
    { // 5. Koga (Fuchsia)
      name: 'Koga', format: 'gen1customgame', aceLevel: 43,
      team: packTeam([
        g1mon('Koffing', ['poisongas', 'tackle', 'smog', 'selfdestruct'], 37),
        g1mon('Muk', ['poisongas', 'disable', 'minimize', 'sludge'], 39),
        g1mon('Koffing', ['poisongas', 'tackle', 'smog', 'selfdestruct'], 37),
        g1mon('Weezing', ['tackle', 'smog', 'explosion', 'smokescreen'], 43),
      ]),
    },
    { // 6. Sabrina (Saffron)
      name: 'Sabrina', format: 'gen1customgame', aceLevel: 43,
      team: packTeam([
        g1mon('Kadabra', ['psychic', 'disable', 'kinesis', 'recover'], 38),
        g1mon('Mr. Mime', ['psybeam', 'barrier', 'psychic', 'substitute'], 37),
        g1mon('Venomoth', ['poisonpowder', 'supersonic', 'psybeam', 'stunspore'], 38),
        g1mon('Alakazam', ['psychic', 'recover', 'kinesis', 'disable'], 43),
      ]),
    },
    { // 7. Blaine (Cinnabar)
      name: 'Blaine', format: 'gen1customgame', aceLevel: 47,
      team: packTeam([
        g1mon('Growlithe', ['ember', 'firespin', 'leer', 'agility'], 42),
        g1mon('Ponyta', ['ember', 'firespin', 'leer', 'agility'], 40),
        g1mon('Rapidash', ['ember', 'stomp', 'firespin', 'agility'], 42),
        g1mon('Arcanine', ['firespin', 'leer', 'roar', 'takedown'], 47),
      ]),
    },
    { // 8. Giovanni (Viridian)
      name: 'Giovanni', format: 'gen1customgame', aceLevel: 50,
      team: packTeam([
        g1mon('Rhyhorn', ['stomp', 'tailwhip', 'furyattack', 'hornattack'], 45),
        g1mon('Dugtrio', ['slash', 'growl', 'dig', 'sandattack'], 42),
        g1mon('Nidoqueen', ['scratch', 'tackle', 'bodyslam', 'poisonsting'], 44),
        g1mon('Nidoking', ['poisonsting', 'leer', 'doublekick', 'thrash'], 45),
        g1mon('Rhydon', ['stomp', 'tailwhip', 'furyattack', 'horndrill'], 50),
      ]),
    },
  ],

  // ─── GEN 2: Cyndaquil line (Gold/Silver) ───────────────────────────────────
  [
    { // 1. Falkner (Violet City)
      name: 'Falkner', format: 'gen2customgame', aceLevel: 9,
      team: packTeam([
        g2mon('Pidgey', null, ['gust', 'sandattack'], 7),
        g2mon('Pidgeotto', null, ['gust', 'sandattack', 'quickattack'], 9),
      ]),
    },
    { // 2. Bugsy (Azalea Town)
      name: 'Bugsy', format: 'gen2customgame', aceLevel: 16,
      team: packTeam([
        g2mon('Metapod', null, ['tackle', 'stringshot', 'harden'], 14),
        g2mon('Kakuna', null, ['poisonsting', 'stringshot', 'harden'], 14),
        g2mon('Scyther', null, ['quickattack', 'leer', 'focusenergy', 'furycutter'], 16),
      ]),
    },
    { // 3. Whitney (Goldenrod City)
      name: 'Whitney', format: 'gen2customgame', aceLevel: 20,
      team: packTeam([
        g2mon('Clefairy', null, ['metronome', 'attract', 'doubleslap', 'encore'], 18),
        g2mon('Miltank', null, ['rollout', 'attract', 'bodyslam', 'milkdrink'], 20),
      ]),
    },
    { // 4. Morty (Ecruteak City)
      name: 'Morty', format: 'gen2customgame', aceLevel: 25,
      team: packTeam([
        g2mon('Gastly', null, ['lick', 'hypnosis', 'meanlook', 'spite'], 21),
        g2mon('Haunter', null, ['lick', 'hypnosis', 'curse', 'meanlook'], 21),
        g2mon('Gengar', null, ['shadowball', 'hypnosis', 'meanlook', 'curse'], 25),
        g2mon('Haunter', null, ['lick', 'hypnosis', 'meanlook', 'nightshade'], 23),
      ]),
    },
    { // 5. Chuck (Cianwood City)
      name: 'Chuck', format: 'gen2customgame', aceLevel: 30,
      team: packTeam([
        g2mon('Primeape', null, ['karatechop', 'furyswipes', 'headbutt', 'lowkick'], 27),
        g2mon('Poliwrath', null, ['surf', 'bodyslam', 'hypnosis', 'dynamicpunch'], 30),
      ]),
    },
    { // 6. Jasmine (Olivine City)
      name: 'Jasmine', format: 'gen2customgame', aceLevel: 35,
      team: packTeam([
        g2mon('Magnemite', null, ['thunderwave', 'sonicboom', 'supersonic', 'thundershock'], 30),
        g2mon('Magnemite', null, ['thunderwave', 'sonicboom', 'supersonic', 'thundershock'], 30),
        g2mon('Steelix', null, ['irontail', 'screech', 'leer', 'sandstorm'], 35),
      ]),
    },
    { // 7. Pryce (Mahogany Town)
      name: 'Pryce', format: 'gen2customgame', aceLevel: 31,
      team: packTeam([
        g2mon('Seel', null, ['aurorabeam', 'rest', 'headbutt', 'icywind'], 27),
        g2mon('Dewgong', null, ['aurorabeam', 'icebeam', 'rest', 'headbutt'], 29),
        g2mon('Piloswine', null, ['blizzard', 'ancientpower', 'mist', 'powdersnow'], 31),
      ]),
    },
    { // 8. Clair (Blackthorn City)
      name: 'Clair', format: 'gen2customgame', aceLevel: 40,
      team: packTeam([
        g2mon('Dragonair', null, ['dragonbreath', 'thunderwave', 'slam', 'hyperbeam'], 37),
        g2mon('Dragonair', null, ['dragonbreath', 'thunderwave', 'slam', 'hyperbeam'], 37),
        g2mon('Dragonair', null, ['dragonbreath', 'thunderwave', 'slam', 'hyperbeam'], 37),
        g2mon('Kingdra', null, ['smokescreen', 'surf', 'leer', 'dragonbreath'], 40),
      ]),
    },
  ],

  // ─── GEN 3: Mudkip line (Ruby/Sapphire) ────────────────────────────────────
  [
    { // 1. Roxanne (Rustboro City)
      name: 'Roxanne', format: 'gen3customgame', aceLevel: 15,
      team: packTeam([
        gXmon('Geodude', null, 'rockhead', ['tackle', 'defensecurl', 'rockthrow', 'magnitude'], '', 14),
        gXmon('Geodude', null, 'rockhead', ['tackle', 'defensecurl', 'rockthrow', 'magnitude'], '', 14),
        gXmon('Nosepass', null, 'sturdy', ['tackle', 'harden', 'rockthrow', 'block'], '', 15),
      ]),
    },
    { // 2. Brawly (Dewford Town)
      name: 'Brawly', format: 'gen3customgame', aceLevel: 18,
      team: packTeam([
        gXmon('Machop', null, 'guts', ['lowkick', 'leer', 'focusenergy', 'karatechop'], '', 17),
        gXmon('Makuhita', null, 'guts', ['tackle', 'sandattack', 'armthrust', 'vitalthrow'], '', 18),
      ]),
    },
    { // 3. Wattson (Mauville City)
      name: 'Wattson', format: 'gen3customgame', aceLevel: 24,
      team: packTeam([
        gXmon('Voltorb', null, 'static', ['sonicboom', 'screech', 'tackle', 'rollout'], '', 20),
        gXmon('Electrike', null, 'static', ['tackle', 'thundershock', 'leer', 'quickattack'], '', 20),
        gXmon('Magneton', null, 'magnetpull', ['thundershock', 'thunderwave', 'sonicboom', 'supersonic'], '', 22),
        gXmon('Manectric', null, 'static', ['thunderbolt', 'thunderwave', 'quickattack', 'roar'], '', 24),
      ]),
    },
    { // 4. Flannery (Lavaridge Town)
      name: 'Flannery', format: 'gen3customgame', aceLevel: 28,
      team: packTeam([
        gXmon('Slugma', null, 'magmaarmor', ['smog', 'ember', 'rockthrow', 'harden'], '', 26),
        gXmon('Slugma', null, 'magmaarmor', ['smog', 'ember', 'rockthrow', 'harden'], '', 26),
        gXmon('Torkoal', null, 'whitesmoke', ['overheat', 'ember', 'smokescreen', 'bodyslam'], '', 28),
      ]),
    },
    { // 5. Norman (Petalburg City)
      name: 'Norman', format: 'gen3customgame', aceLevel: 31,
      team: packTeam([
        gXmon('Spinda', null, 'owntempo', ['psybeam', 'teeterdance', 'encore', 'facade'], '', 27),
        gXmon('Vigoroth', null, 'vitalspirit', ['slash', 'facade', 'faintattack', 'encore'], '', 27),
        gXmon('Linoone', null, 'pickup', ['headbutt', 'slash', 'tailwhip', 'extremespeed'], '', 29),
        gXmon('Slaking', null, 'truant', ['facade', 'secretpower', 'yawn', 'encore'], '', 31),
      ]),
    },
    { // 6. Winona (Fortree City)
      name: 'Winona', format: 'gen3customgame', aceLevel: 33,
      team: packTeam([
        gXmon('Swablu', null, 'naturalcure', ['peck', 'growl', 'astonish', 'sing'], '', 29),
        gXmon('Tropius', null, 'chlorophyll', ['razorleaf', 'leer', 'stomp', 'sweetscent'], '', 29),
        gXmon('Pelipper', null, 'keeneye', ['wingattack', 'watergun', 'supersonic', 'protect'], '', 30),
        gXmon('Skarmory', null, 'keeneye', ['steelwing', 'sandstorm', 'swift', 'agility'], '', 31),
        gXmon('Altaria', null, 'naturalcure', ['dragonbreath', 'sing', 'safeguard', 'aerialace'], '', 33),
      ]),
    },
    { // 7. Tate & Liza (Mossdeep City) — run as single battle
      name: 'Tate & Liza', format: 'gen3customgame', aceLevel: 42,
      team: packTeam([
        gXmon('Solrock', null, 'levitate', ['confusion', 'flamethrower', 'calmmind', 'lightscreen'], '', 42),
        gXmon('Lunatone', null, 'levitate', ['confusion', 'blizzard', 'calmmind', 'reflect'], '', 42),
      ]),
    },
    { // 8. Wallace (Sootopolis City)
      name: 'Wallace', format: 'gen3customgame', aceLevel: 43,
      team: packTeam([
        gXmon('Luvdisc', null, 'swiftswim', ['watergun', 'attract', 'sweetkiss', 'tackle'], '', 40),
        gXmon('Whiscash', null, 'oblivious', ['surf', 'earthquake', 'amnesia', 'rest'], '', 42),
        gXmon('Sealeo', null, 'thickfat', ['watergun', 'aurorabeam', 'icebeam', 'encore'], '', 40),
        gXmon('Seaking', null, 'swiftswim', ['waterfall', 'horndrill', 'furyattack', 'supersonic'], '', 42),
        gXmon('Milotic', null, 'marvelscale', ['surf', 'icebeam', 'recover', 'mirrorcoat'], '', 43),
      ]),
    },
  ],

  // ─── GEN 4: Chimchar line (Diamond/Pearl) ──────────────────────────────────
  [
    { // 1. Roark (Oreburgh City)
      name: 'Roark', format: 'gen4customgame', aceLevel: 14,
      team: packTeam([
        gXmon('Geodude', null, 'rockhead', ['tackle', 'defensecurl', 'rockthrow', 'magnitude'], '', 12),
        gXmon('Onix', null, 'rockhead', ['tackle', 'screech', 'bind', 'rockthrow'], '', 12),
        gXmon('Cranidos', null, 'moldbreaker', ['headbutt', 'leer', 'focusenergy', 'pursuit'], '', 14),
      ]),
    },
    { // 2. Gardenia (Eterna City)
      name: 'Gardenia', format: 'gen4customgame', aceLevel: 22,
      team: packTeam([
        gXmon('Cherubi', null, 'chlorophyll', ['tackle', 'leer', 'growth', 'magicalleaf'], '', 19),
        gXmon('Turtwig', null, 'overgrow', ['tackle', 'growl', 'absorb', 'razorleaf'], '', 19),
        gXmon('Roserade', null, 'naturalcure', ['magicalleaf', 'poisonsting', 'stunspore', 'gigadrain'], '', 22),
      ]),
    },
    { // 3. Maylene (Veilstone City)
      name: 'Maylene', format: 'gen4customgame', aceLevel: 30,
      team: packTeam([
        gXmon('Meditite', null, 'purepower', ['confusion', 'detect', 'forcepalm', 'brickbreak'], '', 27),
        gXmon('Machoke', null, 'guts', ['karatechop', 'seismictoss', 'focusenergy', 'revenge'], '', 27),
        gXmon('Lucario', null, 'steadfast', ['forcepalm', 'quickattack', 'detect', 'metalclaw'], '', 30),
      ]),
    },
    { // 4. Crasher Wake (Pastoria City)
      name: 'Crasher Wake', format: 'gen4customgame', aceLevel: 30,
      team: packTeam([
        gXmon('Gyarados', null, 'intimidate', ['bite', 'dragonrage', 'twister', 'leer'], '', 27),
        gXmon('Quagsire', null, 'waterabsorb', ['watergun', 'mudslap', 'headbutt', 'yawn'], '', 27),
        gXmon('Floatzel', null, 'swiftswim', ['waterpulse', 'crunch', 'pursuit', 'swift'], '', 30),
      ]),
    },
    { // 5. Fantina (Hearthome City)
      name: 'Fantina', format: 'gen4customgame', aceLevel: 36,
      team: packTeam([
        gXmon('Drifblim', null, 'aftermath', ['gust', 'shadowball', 'minimize', 'batonpass'], '', 32),
        gXmon('Gengar', null, 'levitate', ['shadowball', 'hypnosis', 'lick', 'confuseray'], '', 34),
        gXmon('Mismagius', null, 'levitate', ['shadowball', 'confuseray', 'psywave', 'hypnosis'], '', 36),
      ]),
    },
    { // 6. Byron (Canalave City)
      name: 'Byron', format: 'gen4customgame', aceLevel: 39,
      team: packTeam([
        gXmon('Bronzor', null, 'heatproof', ['irondefense', 'safeguard', 'gyroball', 'confusion'], '', 36),
        gXmon('Steelix', null, 'sturdy', ['ironhead', 'earthquake', 'screech', 'dragonbreath'], '', 36),
        gXmon('Bastiodon', null, 'sturdy', ['metalburst', 'ancientpower', 'irondefense', 'protect'], '', 39),
      ]),
    },
    { // 7. Candice (Snowpoint City)
      name: 'Candice', format: 'gen4customgame', aceLevel: 42,
      team: packTeam([
        gXmon('Snover', null, 'snowwarning', ['blizzard', 'razorleaf', 'iceshard', 'ingrain'], '', 38),
        gXmon('Sneasel', null, 'keeneye', ['iceshard', 'slash', 'pursuit', 'faintattack'], '', 38),
        gXmon('Medicham', null, 'purepower', ['brickbreak', 'detect', 'psybeam', 'forcepalm'], '', 40),
        gXmon('Abomasnow', null, 'snowwarning', ['blizzard', 'woodhammer', 'iceshard', 'ingrain'], '', 42),
      ]),
    },
    { // 8. Volkner (Sunyshore City)
      name: 'Volkner', format: 'gen4customgame', aceLevel: 49,
      team: packTeam([
        gXmon('Raichu', null, 'static', ['thunderbolt', 'quickattack', 'focusblast', 'thunderwave'], '', 46),
        gXmon('Ambipom', null, 'technician', ['swift', 'doublehit', 'fakeout', 'return'], '', 47),
        gXmon('Octillery', null, 'sniperability', ['surf', 'icebeam', 'fireblast', 'thunderwave'], '', 47),
        gXmon('Luxray', null, 'intimidate', ['thunderbolt', 'crunch', 'thunderfang', 'scaryface'], '', 49),
      ]),
    },
  ],

  // ─── GEN 5: Tepig line (Black/White) ───────────────────────────────────────
  [
    { // 1. Cress (Striaton City — Water specialist vs Fire starter)
      name: 'Cress', format: 'gen5customgame', aceLevel: 14,
      team: packTeam([
        gXmon('Lillipup', null, 'vitalspirit', ['tackle', 'leer', 'odorsleuth', 'helpinghand'], '', 12),
        gXmon('Panpour', null, 'gluttony', ['scratch', 'leer', 'lick', 'watergun'], '', 14),
      ]),
    },
    { // 2. Lenora (Nacrene City)
      name: 'Lenora', format: 'gen5customgame', aceLevel: 20,
      team: packTeam([
        gXmon('Herdier', null, 'intimidate', ['bite', 'leer', 'headbutt', 'takedown'], '', 18),
        gXmon('Watchog', null, 'illuminate', ['leer', 'lowkick', 'bite', 'retaliate'], '', 20),
      ]),
    },
    { // 3. Burgh (Castelia City)
      name: 'Burgh', format: 'gen5customgame', aceLevel: 24,
      team: packTeam([
        gXmon('Whirlipede', null, 'poisonpoint', ['rollout', 'screech', 'protect', 'poisonsting'], '', 21),
        gXmon('Dwebble', null, 'sturdy', ['rockblast', 'bugbite', 'smackdown', 'protect'], '', 21),
        gXmon('Leavanny', null, 'swarm', ['xscissor', 'razorleaf', 'bugbite', 'stunspore'], '', 24),
      ]),
    },
    { // 4. Elesa (Nimbasa City)
      name: 'Elesa', format: 'gen5customgame', aceLevel: 28,
      team: packTeam([
        gXmon('Emolga', null, 'static', ['acrobatics', 'spark', 'quickattack', 'discharge'], '', 25),
        gXmon('Emolga', null, 'static', ['acrobatics', 'spark', 'quickattack', 'discharge'], '', 25),
        gXmon('Zebstrika', null, 'lightningrod', ['voltswitch', 'pursuit', 'quickattack', 'discharge'], '', 28),
      ]),
    },
    { // 5. Clay (Driftveil City)
      name: 'Clay', format: 'gen5customgame', aceLevel: 31,
      team: packTeam([
        gXmon('Krokorok', null, 'intimidate', ['crunch', 'sandattack', 'assurance', 'bulldoze'], '', 29),
        gXmon('Palpitoad', null, 'waterabsorb', ['mudshot', 'watergun', 'supersonic', 'sludge'], '', 29),
        gXmon('Excadrill', null, 'sandrush', ['slash', 'rockslide', 'metalclaw', 'dig'], '', 31),
      ]),
    },
    { // 6. Skyla (Mistralton City)
      name: 'Skyla', format: 'gen5customgame', aceLevel: 35,
      team: packTeam([
        gXmon('Swoobat', null, 'unaware', ['airslash', 'attract', 'assurance', 'gust'], '', 33),
        gXmon('Unfezant', null, 'bigpecks', ['airslash', 'detect', 'leer', 'quickattack'], '', 33),
        gXmon('Swanna', null, 'keeneye', ['surf', 'airslash', 'roost', 'icebeam'], '', 35),
      ]),
    },
    { // 7. Brycen (Icirrus City)
      name: 'Brycen', format: 'gen5customgame', aceLevel: 39,
      team: packTeam([
        gXmon('Vanillish', null, 'icebody', ['icebeam', 'iceshard', 'icywind', 'acidarmor'], '', 37),
        gXmon('Cryogonal', null, 'levitate', ['icebeam', 'slash', 'reflect', 'aurorabeam'], '', 37),
        gXmon('Beartic', null, 'snowcloak', ['iciclecrash', 'superpower', 'iceshard', 'blizzard'], '', 39),
      ]),
    },
    { // 8. Drayden (Opelucid City)
      name: 'Drayden', format: 'gen5customgame', aceLevel: 43,
      team: packTeam([
        gXmon('Fraxure', null, 'rivalry', ['slash', 'dragondance', 'dualchop', 'dragonrage'], '', 41),
        gXmon('Fraxure', null, 'rivalry', ['slash', 'dragondance', 'dualchop', 'dragonrage'], '', 41),
        gXmon('Haxorus', null, 'rivalry', ['outrage', 'dragondance', 'dualchop', 'slash'], '', 43),
      ]),
    },
  ],
];

// =============================================================================
// STARTER POKEMON DATA
// starterTeam(starterIdx, gymIdx) → packed team string
// gymIdx 0-7 → evolution stage: 0-1 base, 2-4 middle, 5-7 final
// =============================================================================

function getStage(gymIdx) {
  if (gymIdx <= 1) return 0;  // base (gyms 1-2)
  if (gymIdx <= 4) return 1;  // middle (gyms 3-5)
  return 2;                   // final (gyms 6-8)
}

// STARTER_DATA[starterIdx][gymIdx] = {species, moves, nature, gen}
// Level = aceLevel of that gym leader (set at runtime)
const STARTER_DATA = [
  // ─── 0: Bulbasaur line (Gen 1) ──────────────────────────────────────────────
  [
    // Gym 1 (L14, Bulbasaur)
    { species: 'Bulbasaur', moves: ['vinewhip', 'leechseed', 'tackle', 'growl'], nature: '', gen: 1 },
    // Gym 2 (L21, Bulbasaur)
    { species: 'Bulbasaur', moves: ['vinewhip', 'leechseed', 'poisonpowder', 'tackle'], nature: '', gen: 1 },
    // Gym 3 (L24, Ivysaur)
    { species: 'Ivysaur', moves: ['vinewhip', 'leechseed', 'poisonpowder', 'razorleaf'], nature: '', gen: 1 },
    // Gym 4 (L29, Ivysaur)
    { species: 'Ivysaur', moves: ['razorleaf', 'leechseed', 'poisonpowder', 'sleeppowder'], nature: '', gen: 1 },
    // Gym 5 (L43, Ivysaur)
    { species: 'Ivysaur', moves: ['razorleaf', 'sleeppowder', 'poisonpowder', 'growth'], nature: '', gen: 1 },
    // Gym 6 (L43, Venusaur)
    { species: 'Venusaur', moves: ['razorleaf', 'sleeppowder', 'poisonpowder', 'growth'], nature: '', gen: 1 },
    // Gym 7 (L47, Venusaur)
    { species: 'Venusaur', moves: ['razorleaf', 'sleeppowder', 'poisonpowder', 'growth'], nature: '', gen: 1 },
    // Gym 8 (L50, Venusaur)
    { species: 'Venusaur', moves: ['razorleaf', 'sleeppowder', 'poisonpowder', 'growth'], nature: '', gen: 1 },
  ],

  // ─── 1: Cyndaquil line (Gen 2) ──────────────────────────────────────────────
  [
    // Gym 1 (L9, Cyndaquil)
    { species: 'Cyndaquil', moves: ['tackle', 'leer', 'smokescreen', 'ember'], nature: '', gen: 2 },
    // Gym 2 (L16, Cyndaquil)
    { species: 'Cyndaquil', moves: ['tackle', 'ember', 'smokescreen', 'quickattack'], nature: '', gen: 2 },
    // Gym 3 (L20, Quilava)
    { species: 'Quilava', moves: ['ember', 'quickattack', 'flamewheel', 'smokescreen'], nature: '', gen: 2 },
    // Gym 4 (L25, Quilava)
    { species: 'Quilava', moves: ['ember', 'quickattack', 'flamewheel', 'smokescreen'], nature: '', gen: 2 },
    // Gym 5 (L30, Quilava)
    { species: 'Quilava', moves: ['flamewheel', 'quickattack', 'ember', 'smokescreen'], nature: '', gen: 2 },
    // Gym 6 (L35, Typhlosion)
    { species: 'Typhlosion', moves: ['flamewheel', 'quickattack', 'ember', 'smokescreen'], nature: '', gen: 2 },
    // Gym 7 (L31, Typhlosion)
    { species: 'Typhlosion', moves: ['flamewheel', 'quickattack', 'ember', 'smokescreen'], nature: '', gen: 2 },
    // Gym 8 (L40, Typhlosion)
    { species: 'Typhlosion', moves: ['swift', 'flamewheel', 'quickattack', 'ember'], nature: '', gen: 2 },
  ],

  // ─── 2: Mudkip line (Gen 3) ─────────────────────────────────────────────────
  [
    // Gym 1 (L15, Mudkip)
    { species: 'Mudkip', moves: ['tackle', 'growl', 'mudslap', 'watergun'], nature: 'Adamant', gen: 3 },
    // Gym 2 (L18, Mudkip)
    { species: 'Mudkip', moves: ['mudslap', 'watergun', 'bide', 'tackle'], nature: 'Adamant', gen: 3 },
    // Gym 3 (L24, Marshtomp)
    { species: 'Marshtomp', moves: ['watergun', 'mudshot', 'protect', 'tackle'], nature: 'Adamant', gen: 3 },
    // Gym 4 (L28, Marshtomp)
    { species: 'Marshtomp', moves: ['watergun', 'mudshot', 'protect', 'takedown'], nature: 'Adamant', gen: 3 },
    // Gym 5 (L31, Marshtomp)
    { species: 'Marshtomp', moves: ['surf', 'mudshot', 'protect', 'takedown'], nature: 'Adamant', gen: 3 },
    // Gym 6 (L33, Swampert — forced early evolution)
    { species: 'Swampert', moves: ['surf', 'mudshot', 'protect', 'mirrorcoat'], nature: 'Adamant', gen: 3 },
    // Gym 7 (L42, Swampert)
    { species: 'Swampert', moves: ['surf', 'earthquake', 'mudshot', 'protect'], nature: 'Adamant', gen: 3 },
    // Gym 8 (L43, Swampert)
    { species: 'Swampert', moves: ['surf', 'earthquake', 'mudshot', 'protect'], nature: 'Adamant', gen: 3 },
  ],

  // ─── 3: Chimchar line (Gen 4) ───────────────────────────────────────────────
  [
    // Gym 1 (L14, Chimchar)
    { species: 'Chimchar', moves: ['scratch', 'leer', 'ember', 'taunt'], nature: 'Jolly', gen: 4 },
    // Gym 2 (L22, Chimchar — forced, normally would be Monferno at L14)
    { species: 'Chimchar', moves: ['ember', 'taunt', 'furyswipes', 'scratch'], nature: 'Jolly', gen: 4 },
    // Gym 3 (L30, Monferno)
    { species: 'Monferno', moves: ['machpunch', 'furyswipes', 'flamewheel', 'ember'], nature: 'Jolly', gen: 4 },
    // Gym 4 (L30, Monferno)
    { species: 'Monferno', moves: ['machpunch', 'furyswipes', 'flamewheel', 'ember'], nature: 'Jolly', gen: 4 },
    // Gym 5 (L36, Monferno — forced, normally evolves at L36)
    { species: 'Monferno', moves: ['machpunch', 'furyswipes', 'flamewheel', 'ember'], nature: 'Jolly', gen: 4 },
    // Gym 6 (L39, Infernape)
    { species: 'Infernape', moves: ['closecombat', 'machpunch', 'flamewheel', 'flareblitz'], nature: 'Jolly', gen: 4 },
    // Gym 7 (L42, Infernape)
    { species: 'Infernape', moves: ['closecombat', 'machpunch', 'flamewheel', 'flareblitz'], nature: 'Jolly', gen: 4 },
    // Gym 8 (L49, Infernape)
    { species: 'Infernape', moves: ['closecombat', 'machpunch', 'flamewheel', 'flareblitz'], nature: 'Jolly', gen: 4 },
  ],

  // ─── 4: Tepig line (Gen 5) ──────────────────────────────────────────────────
  [
    // Gym 1 (L14, Tepig)
    { species: 'Tepig', moves: ['tackle', 'ember', 'odorsleuth', 'tailwhip'], nature: 'Adamant', gen: 5 },
    // Gym 2 (L20, Tepig — forced base form past natural evolution L17)
    { species: 'Tepig', moves: ['flamecharge', 'ember', 'odorsleuth', 'defensecurl'], nature: 'Adamant', gen: 5 },
    // Gym 3 (L24, Pignite)
    { species: 'Pignite', moves: ['armthrust', 'defensecurl', 'ember', 'tackle'], nature: 'Adamant', gen: 5 },
    // Gym 4 (L28, Pignite)
    { species: 'Pignite', moves: ['armthrust', 'flamecharge', 'defensecurl', 'ember'], nature: 'Adamant', gen: 5 },
    // Gym 5 (L31, Pignite)
    { species: 'Pignite', moves: ['armthrust', 'flamecharge', 'smog', 'defensecurl'], nature: 'Adamant', gen: 5 },
    // Gym 6 (L35, Emboar — forced early evolution; Pignite evolves at L36)
    { species: 'Emboar', moves: ['armthrust', 'flamecharge', 'smog', 'rollout'], nature: 'Adamant', gen: 5 },
    // Gym 7 (L39, Emboar)
    { species: 'Emboar', moves: ['takedown', 'flamecharge', 'armthrust', 'rollout'], nature: 'Adamant', gen: 5 },
    // Gym 8 (L43, Emboar)
    { species: 'Emboar', moves: ['heatcrash', 'takedown', 'flamecharge', 'armthrust'], nature: 'Adamant', gen: 5 },
  ],
];

const STARTER_NAMES = ['Bulbasaur', 'Cyndaquil', 'Mudkip', 'Chimchar', 'Tepig'];
const GEN_NUMBERS = [1, 2, 3, 4, 5];

// Build a starter's packed team string for a given starter/gym
function buildStarterTeam(starterIdx, gymIdx) {
  const gymLeader = GYM_LEADERS[starterIdx][gymIdx];
  const starterInfo = STARTER_DATA[starterIdx][gymIdx];
  const level = gymLeader.aceLevel;
  const gen = starterInfo.gen;

  if (gen <= 2) {
    return g1mon(starterInfo.species, starterInfo.moves, level)
      .replace('|252,252,252,252,252,252|', '|252,252,252,252,252,252|'); // already MAX_EVS
  } else {
    return gXmon(starterInfo.species, null, null, starterInfo.moves, starterInfo.nature, level);
  }
}

// =============================================================================
// BATTLE RUNNER
// Returns { won, koCount, gymTotal, turns }
// =============================================================================
async function runBattle(format, starterTeam, gymTeam, seed, genNum) {
  const battleStream = new BattleStream();
  const streams = getPlayerStreams(battleStream);

  const seedStr = `${seed[0]},${seed[1]},${seed[2]},${seed[3]}`;
  const aiSeed1 = `${seed[0] ^ 0xAAAA},${seed[1]},${seed[2]},${seed[3]}`;
  const aiSeed2 = `${seed[0] ^ 0x5555},${seed[1]},${seed[2]},${seed[3]}`;

  const p1 = new DamageFirstAI(streams.p1, { seed: aiSeed1, move: 1.0 }, genNum);
  const p2 = new RandomPlayerAI(streams.p2, { seed: aiSeed2, move: 0.7 });

  void p1.start();
  void p2.start();

  void streams.omniscient.write(
    `>start ${JSON.stringify({ formatid: format, seed: seedStr })}\n` +
    `>player p1 ${JSON.stringify({ name: 'Starter', team: starterTeam })}\n` +
    `>player p2 ${JSON.stringify({ name: 'GymLeader', team: gymTeam })}`
  );

  let won = false;
  let koCount = 0;
  let turns = 0;
  const gymTotal = gymTeam.split(']').length;

  try {
    for await (const chunk of streams.omniscient) {
      // The omniscient stream yields raw battle protocol lines directly.
      // Each chunk is a newline-delimited set of PS battle log lines.
      for (const line of chunk.split('\n')) {
        if (line.startsWith('|faint|p2')) koCount++;
        if (line.startsWith('|win|')) won = (line.slice(5).trim() === 'Starter');
        if (line.startsWith('|turn|')) {
          const t = parseInt(line.split('|')[2]);
          if (!isNaN(t)) turns = t;
        }
      }
    }
  } catch (err) {
    // Battle crashed — count as a loss for the starter
  }

  try { streams.omniscient.writeEnd(); } catch (_) {}

  return { won, koCount, gymTotal, turns };
}

// =============================================================================
// CONCURRENCY POOL
// =============================================================================
async function runConcurrent(tasks, limit) {
  const results = new Array(tasks.length);
  let idx = 0;

  async function worker() {
    while (idx < tasks.length) {
      const i = idx++;
      results[i] = await tasks[i]();
    }
  }

  const workers = [];
  for (let i = 0; i < Math.min(limit, tasks.length); i++) {
    workers.push(worker());
  }
  await Promise.all(workers);
  return results;
}

// =============================================================================
// MAIN SIMULATION
// =============================================================================
async function runSimulation() {
  // Ensure output dir exists
  if (!fs.existsSync(OUTPUT_DIR)) fs.mkdirSync(OUTPUT_DIR, { recursive: true });

  // Open CSV stream for raw_battles
  const rawStream = fs.createWriteStream(path.join(OUTPUT_DIR, 'raw_battles.csv'));
  rawStream.write('iteration,starter_line,gym_number,battle_run,evolution_stage,gym_leader_name,starter_won,gym_leader_pokemon_defeated,gym_leader_pokemon_total,turns\n');

  // Accumulators for statistics
  // koRatioSums[starterIdx][iterIdx] = sum of 80 KO proportions
  // durationMeans[starterIdx][iterIdx] = mean of 80 turn counts
  // rankingScores[starterIdx][iterIdx] = sum of 8 ranking points
  const koRatioSums = Array.from({ length: 5 }, () => new Float64Array(TOTAL_ITERATIONS));
  const durationMeans = Array.from({ length: 5 }, () => new Float64Array(TOTAL_ITERATIONS));
  const rankingScores = Array.from({ length: 5 }, () => new Float64Array(TOTAL_ITERATIONS));

  // gym_rankings accumulator: [gymIdx][starterIdx] = {totalMetric, count}
  // We need per-gym per-starter average metric per iteration, then rank and sum
  // winRates[gymIdx][starterIdx][iterIdx] = win fraction over 10 battles
  const winRates = Array.from({ length: 8 }, () =>
    Array.from({ length: 5 }, () => new Float64Array(TOTAL_ITERATIONS))
  );
  const koRatios = Array.from({ length: 8 }, () =>
    Array.from({ length: 5 }, () => new Float64Array(TOTAL_ITERATIONS))
  );

  console.log(`Starting simulation: ${TOTAL_ITERATIONS} iterations × 5 starters × 8 gyms × ${BATTLES_PER_GYM} battles = ${TOTAL_ITERATIONS * 5 * 8 * BATTLES_PER_GYM} total battles`);
  console.log(`Concurrency: ${CONCURRENCY}`);
  const startTime = Date.now();

  for (let iter = 0; iter < TOTAL_ITERATIONS; iter++) {
    if (iter % 100 === 0) {
      const elapsed = ((Date.now() - startTime) / 1000).toFixed(1);
      console.log(`Iteration ${iter}/${TOTAL_ITERATIONS} (${elapsed}s elapsed)`);
    }

    // Seed for this iteration — derive per-battle seeds from it
    const baseSeed = [iter + 1, iter * 3 + 7, iter * 7 + 13, iter * 11 + 17];

    // Build all battle tasks for this iteration (5 starters × 8 gyms × 10 battles)
    const tasks = [];
    const taskMeta = []; // {starterIdx, gymIdx, runIdx}

    for (let si = 0; si < 5; si++) {
      for (let gi = 0; gi < 8; gi++) {
        const leader = GYM_LEADERS[si][gi];
        const starterTeam = buildStarterTeam(si, gi);
        const gymTeam = leader.team;
        const format = leader.format;
        const genNum = GEN_NUMBERS[si];

        for (let ri = 0; ri < BATTLES_PER_GYM; ri++) {
          // Unique seed per battle
          const seed = [
            (baseSeed[0] + si * 1000 + gi * 100 + ri * 10) & 0xFFFF,
            (baseSeed[1] + si * 1001 + gi * 101 + ri * 11) & 0xFFFF,
            (baseSeed[2] + si * 1003 + gi * 103 + ri * 13) & 0xFFFF,
            (baseSeed[3] + si * 1007 + gi * 107 + ri * 17) & 0xFFFF,
          ];

          const capturedSi = si;
          const capturedGi = gi;
          const capturedRi = ri;
          const capturedSeed = seed;
          const capturedStarterTeam = starterTeam;
          const capturedGymTeam = gymTeam;
          const capturedFormat = format;
          const capturedGenNum = genNum;

          tasks.push(() => runBattle(capturedFormat, capturedStarterTeam, capturedGymTeam, capturedSeed, capturedGenNum));
          taskMeta.push({ starterIdx: capturedSi, gymIdx: capturedGi, runIdx: capturedRi });
        }
      }
    }

    // Run all battles for this iteration concurrently
    const results = await runConcurrent(tasks, CONCURRENCY);

    // Accumulate results
    // Temporary storage for this iteration: [si][gi] = {wins, koRatioSum, turnsSum}
    const iterData = Array.from({ length: 5 }, () =>
      Array.from({ length: 8 }, () => ({ wins: 0, koRatioSum: 0, turnsSum: 0 }))
    );

    for (let t = 0; t < results.length; t++) {
      const { won, koCount, gymTotal, turns } = results[t];
      const { starterIdx: si, gymIdx: gi, runIdx: ri } = taskMeta[t];
      const leader = GYM_LEADERS[si][gi];
      const stage = getStage(gi);
      const stageNames = ['Base', 'Middle', 'Final'];
      const starterInfo = STARTER_DATA[si][gi];

      // Write to CSV
      rawStream.write(
        `${iter + 1},${STARTER_NAMES[si]},${gi + 1},${ri + 1},${stageNames[stage]},` +
        `${leader.name},${won ? 1 : 0},${koCount},${gymTotal},${turns}\n`
      );

      // Accumulate for stats
      const proportion = gymTotal > 0 ? koCount / gymTotal : 0;
      iterData[si][gi].wins += won ? 1 : 0;
      iterData[si][gi].koRatioSum += proportion;
      iterData[si][gi].turnsSum += turns;
    }

    // Compute per-iteration stats
    for (let si = 0; si < 5; si++) {
      let totalKoRatioSum = 0;
      let totalTurns = 0;

      for (let gi = 0; gi < 8; gi++) {
        const d = iterData[si][gi];
        totalKoRatioSum += d.koRatioSum;
        totalTurns += d.turnsSum;

        // Win rate for ranking
        winRates[gi][si][iter] = d.wins / BATTLES_PER_GYM;
        koRatios[gi][si][iter] = d.koRatioSum / BATTLES_PER_GYM;
      }

      koRatioSums[si][iter] = totalKoRatioSum;
      durationMeans[si][iter] = totalTurns / (8 * BATTLES_PER_GYM);
    }

    // Compute ranking points for this iteration
    for (let gi = 0; gi < 8; gi++) {
      // Rank all 5 starters by win rate (or KO ratio if no one wins)
      const anyWins = Array.from({ length: 5 }, (_, si) => winRates[gi][si][iter]).some(w => w > 0);
      const metric = Array.from({ length: 5 }, (_, si) =>
        anyWins ? winRates[gi][si][iter] : koRatios[gi][si][iter]
      );

      // Sort descending and assign points (5 = best, 1 = worst)
      const ranked = [...metric.map((m, i) => ({ m, i }))].sort((a, b) => b.m - a.m);
      let lastM = null;
      let lastPts = 5;
      for (let r = 0; r < ranked.length; r++) {
        if (ranked[r].m !== lastM) {
          // New distinct performance value
          lastPts = 5 - r;
          lastM = ranked[r].m;
        }
        rankingScores[ranked[r].i][iter] += lastPts;
      }
    }
  }

  rawStream.end();
  console.log('\nSimulation complete. Computing statistics...');

  // ==========================================================================
  // STATISTICS
  // ==========================================================================

  // summary_stats.csv
  const summaryRows = [];
  for (let si = 0; si < 5; si++) {
    const avgKoRatioSum = mean(koRatioSums[si]);
    const avgDuration = mean(durationMeans[si]);
    const avgRankingScore = mean(rankingScores[si]);
    summaryRows.push({
      starter_line: STARTER_NAMES[si],
      avg_ko_ratio_sum: avgKoRatioSum.toFixed(6),
      avg_battle_duration_turns: avgDuration.toFixed(4),
      avg_ranking_score: avgRankingScore.toFixed(4),
    });
  }

  let summaryCsv = 'starter_line,avg_ko_ratio_sum,avg_battle_duration_turns,avg_ranking_score\n';
  for (const r of summaryRows) {
    summaryCsv += `${r.starter_line},${r.avg_ko_ratio_sum},${r.avg_battle_duration_turns},${r.avg_ranking_score}\n`;
  }
  fs.writeFileSync(path.join(OUTPUT_DIR, 'summary_stats.csv'), summaryCsv);

  // gym_rankings.csv
  // Per gym: avg_performance_metric = mean win rate (or KO ratio), points_awarded = avg points
  // We compute per-gym, per-starter: mean metric across iterations, then rank
  const gymRankingRows = [];
  for (let gi = 0; gi < 8; gi++) {
    const avgMetrics = [];
    for (let si = 0; si < 5; si++) {
      const avgWin = mean(winRates[gi][si]);
      const avgKo = mean(koRatios[gi][si]);
      const anyWins = avgWin > 0;
      avgMetrics.push({ si, metric: anyWins ? avgWin : avgKo });
    }

    // Rank and assign points
    const sorted = [...avgMetrics].sort((a, b) => b.metric - a.metric);
    const pointsMap = new Map();
    let lastM = null;
    let lastPts = 5;
    for (let r = 0; r < sorted.length; r++) {
      if (sorted[r].metric !== lastM) {
        lastPts = 5 - r;
        lastM = sorted[r].metric;
      }
      pointsMap.set(sorted[r].si, lastPts);
    }

    for (const { si, metric } of avgMetrics) {
      gymRankingRows.push({
        gym_number: gi + 1,
        starter_line: STARTER_NAMES[si],
        avg_performance_metric: metric.toFixed(6),
        points_awarded: pointsMap.get(si),
      });
    }
  }

  let gymRankingCsv = 'gym_number,starter_line,avg_performance_metric,points_awarded\n';
  for (const r of gymRankingRows) {
    gymRankingCsv += `${r.gym_number},${r.starter_line},${r.avg_performance_metric},${r.points_awarded}\n`;
  }
  fs.writeFileSync(path.join(OUTPUT_DIR, 'gym_rankings.csv'), gymRankingCsv);

  const elapsed = ((Date.now() - startTime) / 1000).toFixed(1);
  console.log(`\nDone in ${elapsed}s.`);
  console.log(`Output files in: ${OUTPUT_DIR}`);
  console.log('  raw_battles.csv');
  console.log('  summary_stats.csv');
  console.log('  gym_rankings.csv');

  console.log('\n=== SUMMARY RESULTS ===');
  console.log('Starter'.padEnd(12), 'Avg KO Ratio Sum'.padEnd(20), 'Avg Duration'.padEnd(16), 'Avg Ranking Score');
  for (const r of summaryRows) {
    console.log(
      r.starter_line.padEnd(12),
      r.avg_ko_ratio_sum.padEnd(20),
      r.avg_battle_duration_turns.padEnd(16),
      r.avg_ranking_score
    );
  }
}

function mean(arr) {
  if (arr.length === 0) return 0;
  let s = 0;
  for (let i = 0; i < arr.length; i++) s += arr[i];
  return s / arr.length;
}

// =============================================================================
// ENTRY POINT
// =============================================================================
runSimulation().catch(err => {
  console.error('Simulation failed:', err);
  process.exit(1);
});
