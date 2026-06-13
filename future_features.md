# Civilitas — Game Roadmap

## Vision

**Civilitas** is a browser-based god game and procedural history generator.

Game theory is the engine underneath — Hawk/Dove strategies, memory, reputation, tit-for-tat. But what the player actually *experiences* is watching civilizations rise, fracture, wage war, survive plagues, and write history. The player is a god who can watch, nudge, or devastate the world at will.

The core loop: **Configure → Watch → Intervene → Read the history it wrote.**

---

## What's Already Built (v0.1 — The Engine)

- Hawk/Dove combat with 5 strategies: Aggressive, Defensive, Cooperative (TfT), Grudger, Detective
- Entities with memory — they remember who betrayed them
- Territory capture with border vs. interior distinction
- Titles & dynasty system: Peasant → Commander / Minister → King
- Bloodline tribes that can break off into Cults (rebellion mechanic)
- Seasons (Spring/Summer/Autumn/Winter) affecting food and survival pressure
- Disease / plague spread
- Reproduction with gender system
- Day/night cycle rendering
- Population graph + World History Log
- Live stats panel

---

## Phase 1 — "The God Feels Real" (Core Interactivity)

Make the player feel like a god. These features turn a passive simulation into a game.

### 1.1 God Hand
- **Click to spawn food** — save a starving civilization, create a strategic advantage
- **Click to smite** — click any entity to instantly kill it (useful for taking out a dominant King)
- **Draw walls** — click-drag to place terrain barriers that physically block movement and combat
- Right-click context menu on entities: Smite / Infect / Bless (+HP)

### 1.2 Hover Tooltips
Hovering any dot shows a glassmorphic popup with:
- Title (King / Commander / Minister / Peasant / Child)
- Age, HP, Kill Count, Food Eaten
- Memory Log — who they remember fighting and whether it was Hawk or Dove
- Tribe ID and bloodline lineage

### 1.3 Third Society Slot
Add a third configurable society in the setup panel (Society Gamma). Three-way conflicts produce balance-of-power dynamics that two societies can't — temporary truces, kingmaker dynamics, dogpiling the strong.

---

## Phase 2 — "The World Has Depth" (Emergent Complexity)

### 2.1 Strategy Evolution (Genetic Drift)
Children inherit their parent society's strategy but with a small mutation chance (e.g., 5% chance of shifting one step on the strategy spectrum). Over generations, the effective strategy of a civilization drifts. A Cooperative society can gradually radicalize into a Grudger society after repeated betrayal. This is real evolutionary game theory.

### 2.2 Non-Zero-Sum Economics (Trade & Alliances)
- Introduce a second resource: **Gold** (produced by Ministers, consumed by Kings to maintain armies)
- Dove-Dove meetings between different societies can trigger **Trade** instead of just bouncing away — both gain a small resource bonus
- Societies can form **Alliances**: shared enemy detection, no combat between allies, joint food territories
- Alliance betrayal triggers Grudger-style memory cascade across the entire betrayed society

### 2.3 Terrain & Biomes
- **Mountains** — impassable barriers that naturally form borders and chokepoints
- **Rivers** — slow movement, but nearby land produces 2x food (fertile valleys)
- **Ruins** — neutral sites that grant large one-time food bonuses; civilizations fight over them
- Terrain is procedurally generated at sim start based on a seed

---

## Phase 3 — "The Game Tells Stories" (Narrative Layer)

### 3.1 Export History Book
After a simulation ends, generate a readable chronicle:

> *"Year 1 — The Alpha and Beta peoples emerged on opposite shores. Year 23 — A plague swept through Beta, killing half their warriors. Year 47 — The Cult of Tribe-7 rose in rebellion, seizing the eastern ruins. By Year 100, Civilitas had three kingdoms and one burning empire."*

The history log already tracks every event. This feature formats it into a narrative document (downloadable as .txt or .md).

### 3.2 Shareable Seeds
Every simulation has a seed that determines starting positions, terrain, and RNG. Seeds are encoded in the URL so players can share specific scenarios:
`civilitas.app/?seed=4f3a9b&years=200`

### 3.3 Named Historical Figures
When a King dies, their name is permanently recorded. Kings get procedurally generated names (e.g., "King Valdur the Grudger" or "Queen Seya the Cooperative"). The history book uses these names.

---

## Phase 4 — "This Is a Real Game" (Polish & Distribution)

### 4.1 Interactive Tutorial
An in-game tutorial that walks new players through:
- What Hawk and Dove mean
- Why Tit-for-Tat wins in the long run
- How to read the territory map
- How to use the God Hand

### 4.2 Scenarios / Presets
Pre-configured starting conditions with dramatic setups:
- **"The Cold War"** — Two Grudger superpowers at maximum aggression, equal size
- **"The Plague Year"** — Patient Zero spawns in Year 1 with 3x spread rate
- **"The Underdog"** — Alpha has 4 entities vs Beta's 32, but Alpha plays Detective
- **"Three Kingdoms"** — Three societies, each with different strategies

### 4.3 Web Deployment
- Deploy to Vercel or Cloudflare Pages under a proper domain
- Mobile-friendly layout (touch events for God Hand)
- OpenGraph preview image for sharing history books on social media

---

---

## RL Mode — Implementation Notes (Paused, Code Is In)

RL mode is fully implemented and lives behind the "Enable RL Mode" checkbox in the setup panel. Code is production-ready but turned off by default. To resume:

### Architecture
- **One `RLBrain` per society** (shared policy — all entities in a society feed one replay buffer and train one network). This is more efficient than per-entity networks and produces emergent collective behaviour.
- **Two networks per brain**: `moveNet` (controls 8-direction movement) and `combatNet` (hawk vs dove decisions).
- **Topology**: `moveNet` = 11→24→16→8, `combatNet` = 11→16→12→2. ReLU hidden, linear output (Q-values).
- **Algorithm**: DQN-lite — experience replay buffer (max 2000), epsilon-greedy exploration (ε=1.0 → 0.05), MSE loss, vanilla SGD, batch size 32 trained every 30 frames.
- **No external dependencies** — pure vanilla JS with `Float32Array` for performance.

### State vector (11 features per entity)
| Index | Feature |
|-------|---------|
| 0 | Own HP (normalized) |
| 1 | Own age (normalized) |
| 2 | Nearest food distance (within 350px scan radius) |
| 3,4 | Nearest food direction (dx, dy unit vector) |
| 5 | Nearest enemy distance |
| 6,7 | Nearest enemy direction |
| 8 | Nearest enemy HP |
| 9 | On own territory (0 or 1) |
| 10 | Nearest friendly distance |

### Reward signals
- **+1.5** for eating food
- **+/- (ΔHP × 0.5)** per frame (survival pressure)
- **+2.0** kill bonus (combat net)
- **-2.0** on death (terminal experience)

### Cult inheritance
When a tribe rebellion spawns a new cult, it creates `new RLBrain(parentBrain)` — the child network clones parent weights and mutates them (8% per-weight chance, ±0.2 strength). The child's epsilon resets to at least 0.35 so it re-explores its mutated policy.

### What to do next when resuming
1. Run the simulation in RL mode for 50+ years at 10x speed to let epsilon decay and policy converge
2. Watch for emergent strategy shifts — aggressive societies should learn hawk in contested zones, dove near own territory
3. Consider adding a reward for territory capture (+0.5 per newly claimed cell)
4. Consider per-entity networks for diversity, but profile performance first
5. Potential extension: add a second hidden layer to `combatNet` for more complex opponent modelling

---

## Immediate Next Steps (What We're Building Now)

1. **Hover tooltips** — highest impact, lowest effort
2. **God Hand: food spawn + smite** — makes it feel like a game immediately
3. **Third society slot in setup UI** — trivial code change, massive gameplay impact
4. **Shareable seeds** — enables sharing, costs almost nothing to build
5. **Strategy mutation on reproduction** — one line of code, unlocks real evolutionary dynamics
