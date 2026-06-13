import './style.css'

// --- SIMULATION CONSTANTS ---
const CONFIG = {
  foodSpawnRate: 4,
  maxFood: 400,
  startPopulation: 16,
  reproductionCost: 50,
  maxHp: 100,
  maxAge: 4800,
  adultAge: 960,
  foodEnergy: 40,
  baseDamage: 8,
  gridSize: 30,
}

// --- REINFORCEMENT LEARNING ---

const RL_STATE_SIZE = 11
const RL_MOVE_ACTIONS = 8
const RL_COMBAT_ACTIONS = 2
// Normalized direction vectors for 8 movement actions (N, NE, E, SE, S, SW, W, NW)
const MOVE_DIRS = [
  [0, -1], [0.707, -0.707], [1, 0], [0.707, 0.707],
  [0, 1], [-0.707, 0.707], [-1, 0], [-0.707, -0.707]
]

class NeuralNet {
  constructor(layerSizes) {
    this.layers = layerSizes
    this.weights = []
    this.biases = []
    for (let l = 0; l < layerSizes.length - 1; l++) {
      const inSize = layerSizes[l]
      const outSize = layerSizes[l + 1]
      const scale = Math.sqrt(2.0 / inSize) // He init
      const w = []
      for (let j = 0; j < outSize; j++) {
        w.push(Float32Array.from({ length: inSize }, () => (Math.random() - 0.5) * 2 * scale))
      }
      this.weights.push(w)
      this.biases.push(new Float32Array(outSize))
    }
  }

  forward(input) {
    const activations = [input instanceof Float32Array ? input : Float32Array.from(input)]
    for (let l = 0; l < this.weights.length; l++) {
      const prev = activations[l]
      const w = this.weights[l]
      const b = this.biases[l]
      const isLast = l === this.weights.length - 1
      const next = new Float32Array(w.length)
      for (let j = 0; j < w.length; j++) {
        let sum = b[j]
        const wj = w[j]
        for (let i = 0; i < prev.length; i++) sum += wj[i] * prev[i]
        next[j] = isLast ? sum : (sum > 0 ? sum : 0) // linear output, ReLU hidden
      }
      activations.push(next)
    }
    return activations
  }

  // MSE loss backprop, SGD update
  backward(activations, target, lr) {
    const L = this.weights.length
    const deltas = new Array(L + 1).fill(null)

    const out = activations[L]
    const outDelta = new Float32Array(out.length)
    for (let j = 0; j < out.length; j++) outDelta[j] = out[j] - target[j]
    deltas[L] = outDelta

    for (let l = L - 1; l >= 1; l--) {
      const act = activations[l]
      const dNext = deltas[l + 1]
      const wNext = this.weights[l]
      const d = new Float32Array(act.length)
      for (let i = 0; i < act.length; i++) {
        if (act[i] <= 0) continue // ReLU derivative
        let err = 0
        for (let j = 0; j < dNext.length; j++) err += dNext[j] * wNext[j][i]
        d[i] = err
      }
      deltas[l] = d
    }

    for (let l = 0; l < L; l++) {
      const dNext = deltas[l + 1]
      const act = activations[l]
      const w = this.weights[l]
      const b = this.biases[l]
      for (let j = 0; j < w.length; j++) {
        const dj = dNext[j]
        if (dj === 0) continue
        const wj = w[j]
        for (let i = 0; i < wj.length; i++) wj[i] -= lr * dj * act[i]
        b[j] -= lr * dj
      }
    }
  }

  clone() {
    const copy = new NeuralNet([...this.layers])
    for (let l = 0; l < this.weights.length; l++) {
      copy.weights[l] = this.weights[l].map(row => new Float32Array(row))
      copy.biases[l] = new Float32Array(this.biases[l])
    }
    return copy
  }

  mutate(rate = 0.08, strength = 0.2) {
    for (const layer of this.weights) {
      for (const row of layer) {
        for (let i = 0; i < row.length; i++) {
          if (Math.random() < rate) row[i] += (Math.random() - 0.5) * 2 * strength
        }
      }
    }
  }
}

// One RLBrain per society — shared policy, all entities feed its replay buffer
class RLBrain {
  constructor(parentBrain = null) {
    this.moveNet = parentBrain
      ? parentBrain.moveNet.clone()
      : new NeuralNet([RL_STATE_SIZE, 24, 16, RL_MOVE_ACTIONS])
    this.combatNet = parentBrain
      ? parentBrain.combatNet.clone()
      : new NeuralNet([RL_STATE_SIZE, 16, 12, RL_COMBAT_ACTIONS])

    if (parentBrain) {
      this.moveNet.mutate(0.1, 0.2)
      this.combatNet.mutate(0.1, 0.2)
    }

    this.moveBuffer = []
    this.combatBuffer = []
    this.maxBuffer = 2000
    // Start fully exploratory; cults inherit parent epsilon so they're less random
    this.epsilon = parentBrain ? Math.max(0.35, parentBrain.epsilon) : 1.0
    this.epsilonDecay = 0.9995
    this.epsilonMin = 0.05
    this.lr = 0.003
    this.gamma = 0.95
    this.totalReward = 0
    this.rewardCount = 0
  }

  chooseMove(state) {
    if (Math.random() < this.epsilon) return Math.floor(Math.random() * RL_MOVE_ACTIONS)
    const acts = this.moveNet.forward(state)
    const q = acts[acts.length - 1]
    let best = 0
    for (let i = 1; i < q.length; i++) if (q[i] > q[best]) best = i
    return best
  }

  chooseCombat(state) {
    if (Math.random() < this.epsilon) return Math.random() < 0.5 ? 0 : 1
    const acts = this.combatNet.forward(state)
    const q = acts[acts.length - 1]
    return q[0] >= q[1] ? 0 : 1 // 0=hawk, 1=dove
  }

  rememberMove(state, action, reward, nextState) {
    this.totalReward += reward
    this.rewardCount++
    if (this.moveBuffer.length >= this.maxBuffer) this.moveBuffer.shift()
    this.moveBuffer.push({ state, action, reward, nextState })
  }

  rememberCombat(state, action, reward, nextState) {
    if (this.combatBuffer.length >= this.maxBuffer) this.combatBuffer.shift()
    this.combatBuffer.push({ state, action, reward, nextState })
  }

  train(batchSize = 32) {
    this._trainNet(this.moveNet, this.moveBuffer, batchSize, RL_MOVE_ACTIONS)
    if (this.combatBuffer.length >= 8) {
      this._trainNet(this.combatNet, this.combatBuffer, Math.min(16, this.combatBuffer.length), RL_COMBAT_ACTIONS)
    }
    this.epsilon = Math.max(this.epsilonMin, this.epsilon * this.epsilonDecay)
  }

  _trainNet(net, buffer, batchSize, numActions) {
    if (buffer.length < batchSize) return
    for (let b = 0; b < batchSize; b++) {
      const exp = buffer[Math.floor(Math.random() * buffer.length)]
      const curActs = net.forward(exp.state)
      const nxtActs = net.forward(exp.nextState)
      const curQ = curActs[curActs.length - 1]
      const nxtQ = nxtActs[nxtActs.length - 1]
      let maxNxt = nxtQ[0]
      for (let i = 1; i < nxtQ.length; i++) if (nxtQ[i] > maxNxt) maxNxt = nxtQ[i]
      const target = new Float32Array(numActions)
      for (let i = 0; i < numActions; i++) target[i] = curQ[i]
      target[exp.action] = exp.reward + this.gamma * maxNxt
      net.backward(curActs, target, this.lr)
    }
  }

  get avgReward() {
    return this.rewardCount > 0 ? (this.totalReward / this.rewardCount).toFixed(3) : '—'
  }
}

// --- DOM ELEMENTS ---
const simContainer = document.getElementById('simulation-container')
const simStage = document.getElementById('simulation-stage')
const graphStage = document.getElementById('graph-stage')
const logStage = document.getElementById('log-stage')
const canvas = document.getElementById('simulation-canvas')
const ctx = canvas.getContext('2d')

const setupPanel = document.getElementById('setup-panel')
const statsPanel = document.getElementById('stats-panel')
const statsContent = document.getElementById('stats-content')
const startBtn = document.getElementById('start-btn')
const resetBtn = document.getElementById('reset-btn')
const pauseBtn = document.getElementById('pause-btn')
const speedControl = document.getElementById('sim-speed-control')
const speedLabel = document.getElementById('speed-label')
const seasonDisplay = document.getElementById('season-display')
const historyLog = document.getElementById('history-log')
const historyLogLarge = document.getElementById('history-log-large')
const showMapBtn = document.getElementById('show-map-btn')
const showGraphBtn = document.getElementById('show-graph-btn')
const showLogBtn = document.getElementById('show-log-btn')

// Graph setup
const graphCanvas = document.getElementById('graph-canvas')
const gCtx = graphCanvas.getContext('2d')
const graphCanvasLarge = document.getElementById('graph-canvas-large')
const gLargeCtx = graphCanvasLarge.getContext('2d')

// Sidebar sim preview (shown when graph view is active)
const simPreviewCanvas = document.getElementById('sim-preview-canvas')
const simPreviewCtx = simPreviewCanvas.getContext('2d')
const simSidebarPreview = document.getElementById('sim-sidebar-preview')
const graphPreviewPanel = document.getElementById('graph-preview-panel')

// --- STATE ---
let width = simStage.clientWidth
let height = simStage.clientHeight
let animationId = null
let isRunning = false
let isPaused = false
let frames = 0
let currentYear = 0
let targetYears = 100
let simSpeed = 1
let activeView = 'map'
let graphHover = null
let mapWidth = width
let mapHeight = height
let rlMode = false
const FRAMES_PER_YEAR = 240

// Seasons
const SEASONS = [
  { name: 'Spring', color: '#8bc34a', foodMult: 1.2, coldDmg: 0 },
  { name: 'Summer', color: '#ffeb3b', foodMult: 2.0, coldDmg: 0 },
  { name: 'Autumn', color: '#ff9800', foodMult: 1.0, coldDmg: 0 },
  { name: 'Winter', color: '#03a9f4', foodMult: 0.3, coldDmg: 0.03 } // Winter is survivable now
]
let currentSeasonIdx = 0

canvas.width = width
canvas.height = height

function updateMapAspect() {
  if (mapWidth > 0 && mapHeight > 0) {
    simContainer.style.setProperty('--map-aspect', `${mapWidth} / ${mapHeight}`)
  }
}

function resizeSimPreview() {
  const pw = simSidebarPreview.clientWidth || 330
  simPreviewCanvas.width = pw
  simPreviewCanvas.height = Math.round(pw * Math.max(1, mapHeight) / Math.max(1, mapWidth))
}

function resizeCanvases() {
  if (activeView === 'map') {
    mapWidth = simStage.clientWidth
    mapHeight = simStage.clientHeight
    width = mapWidth
    height = mapHeight
  } else {
    width = mapWidth
    height = mapHeight
  }
  canvas.width = width
  canvas.height = height
  graphCanvasLarge.width = graphStage.clientWidth
  graphCanvasLarge.height = graphStage.clientHeight
  if (activeView === 'graph') resizeSimPreview()
  if (activeView === 'map') updateMapAspect()
  simulation.drawGraph()
  simulation.draw()
}

window.addEventListener('resize', resizeCanvases)

// --- MATH & UTILS ---
class Vector {
  constructor(x, y) { this.x = x; this.y = y; }
  add(v) { this.x += v.x; this.y += v.y; }
  sub(v) { return new Vector(this.x - v.x, this.y - v.y); }
  mag() { return Math.sqrt(this.x ** 2 + this.y ** 2); }
  normalize() {
    const m = this.mag();
    if (m > 0) { this.x /= m; this.y /= m; }
  }
  mult(n) { this.x *= n; this.y *= n; }
}

function getRandomColor() {
  const h = Math.floor(Math.random() * 360)
  return `hsl(${h}, 100%, 60%)`
}

function logEvent(msg, color = '#ccc') {
  const appendEntry = (target) => {
    const el = document.createElement('div')
    el.style.marginBottom = '6px'
    el.style.borderLeft = `3px solid ${color}`
    el.style.paddingLeft = '6px'
    el.innerHTML = `<strong style="color: #fff;">[Year ${currentYear}]</strong> <span style="color: ${color}">${msg}</span>`
    target.appendChild(el)
    target.scrollTop = target.scrollHeight
  }

  appendEntry(historyLog)
  appendEntry(historyLogLarge)
}

function setActiveView(view) {
  if (activeView === 'map' && view !== 'map') updateMapAspect()
  activeView = view
  simContainer.dataset.view = view

  graphStage.classList.toggle('hidden', view !== 'graph')
  logStage.classList.toggle('hidden', view !== 'log')
  simStage.classList.toggle('active', view === 'map')
  graphStage.classList.toggle('active', view === 'graph')
  logStage.classList.toggle('active', view === 'log')

  showMapBtn.classList.toggle('active', view === 'map')
  showGraphBtn.classList.toggle('active', view === 'graph')
  showLogBtn.classList.toggle('active', view === 'log')

  // In graph view: show sim preview in sidebar, hide mini graph canvas (redundant)
  const isGraph = view === 'graph'
  simSidebarPreview.classList.toggle('hidden', !isGraph)
  graphPreviewPanel.classList.toggle('hidden', isGraph)

  requestAnimationFrame(resizeCanvases)
}

function drawPopulationGraph(targetCanvas, targetCtx) {
  targetCtx.clearRect(0, 0, targetCanvas.width, targetCanvas.height)
  if (simulation.populationHistory.length < 2) return

  const padding = targetCanvas === graphCanvasLarge ? 32 : 10
  const topPadding = targetCanvas === graphCanvasLarge ? 54 : padding
  const bottomPadding = targetCanvas === graphCanvasLarge ? 42 : padding
  const innerW = targetCanvas.width - padding * 2
  const innerH = targetCanvas.height - topPadding - bottomPadding

  let maxPop = 10
  simulation.populationHistory.forEach(record => {
    Object.values(record.data).forEach(s => {
      if (s.pop > maxPop) maxPop = s.pop
    })
  })

  if (targetCanvas === graphCanvasLarge) {
    targetCtx.fillStyle = 'rgba(0,0,0,0.28)'
    targetCtx.fillRect(0, 0, targetCanvas.width, targetCanvas.height)
    targetCtx.strokeStyle = 'rgba(255,255,255,0.08)'
    targetCtx.lineWidth = 1
    for (let i = 0; i <= 4; i++) {
      const y = topPadding + (innerH / 4) * i
      targetCtx.beginPath()
      targetCtx.moveTo(padding, y)
      targetCtx.lineTo(padding + innerW, y)
      targetCtx.stroke()
    }
  }

  simulation.societies.forEach(soc => {
    targetCtx.beginPath()
    targetCtx.strokeStyle = soc.color
    targetCtx.lineWidth = targetCanvas === graphCanvasLarge ? 3 : 2
    
    simulation.populationHistory.forEach((record, index) => {
      const pop = record.data[soc.id] ? record.data[soc.id].pop : 0
      const x = padding + (index / Math.max(1, simulation.populationHistory.length - 1)) * innerW
      const y = topPadding + innerH - (pop / maxPop) * innerH
      
      if (index === 0) targetCtx.moveTo(x, y)
      else targetCtx.lineTo(x, y)
    })
    targetCtx.stroke()
  })

  if (targetCanvas === graphCanvasLarge) {
    if (graphHover !== null) {
      const index = Math.max(0, Math.min(simulation.populationHistory.length - 1, graphHover))
      const record = simulation.populationHistory[index]
      const x = padding + (index / Math.max(1, simulation.populationHistory.length - 1)) * innerW
      const values = simulation.societies
        .map(soc => ({
          society: soc,
          pop: record.data[soc.id] ? record.data[soc.id].pop : 0
        }))
        .filter(item => item.pop > 0)

      targetCtx.strokeStyle = 'rgba(255,255,255,0.35)'
      targetCtx.lineWidth = 1
      targetCtx.beginPath()
      targetCtx.moveTo(x, topPadding)
      targetCtx.lineTo(x, topPadding + innerH)
      targetCtx.stroke()

      values.forEach(item => {
        const y = topPadding + innerH - (item.pop / maxPop) * innerH
        targetCtx.fillStyle = item.society.color
        targetCtx.beginPath()
        targetCtx.arc(x, y, 5, 0, Math.PI * 2)
        targetCtx.fill()
      })

      const boxW = 220
      const boxH = 34 + values.length * 22
      const boxX = Math.min(targetCanvas.width - boxW - 16, Math.max(16, x + 14))
      const boxY = 16
      targetCtx.fillStyle = 'rgba(12,14,20,0.9)'
      targetCtx.strokeStyle = 'rgba(255,255,255,0.16)'
      targetCtx.lineWidth = 1
      targetCtx.beginPath()
      targetCtx.roundRect(boxX, boxY, boxW, boxH, 8)
      targetCtx.fill()
      targetCtx.stroke()

      targetCtx.fillStyle = '#ffffff'
      targetCtx.font = '600 14px Inter, sans-serif'
      targetCtx.fillText(`Year ${record.year}`, boxX + 12, boxY + 22)
      values.forEach((item, itemIndex) => {
        targetCtx.fillStyle = item.society.color
        targetCtx.font = '500 13px Inter, sans-serif'
        targetCtx.fillText(`${item.society.name}: ${item.pop}`, boxX + 12, boxY + 48 + itemIndex * 22)
      })
    }

    targetCtx.fillStyle = '#ffffff'
    targetCtx.font = '600 18px Inter, sans-serif'
    targetCtx.fillText('Population History', padding, 28)
  }
}

// --- CLASSES ---

class Society {
  constructor(id, name, color, strategy, aggression, speed, allowWomenFighters = false) {
    this.id = id
    this.name = name
    this.color = color
    this.strategy = strategy
    this.aggression = aggression / 100
    this.speed = speed
    this.allowWomenFighters = allowWomenFighters
    this.king = null
    this.brain = null // RLBrain instance, set in simulation.init() when rlMode is on
  }
}

class Food {
  constructor(x, y) {
    this.pos = new Vector(x, y)
    this.size = 2
    this.color = '#00ff88'
  }
  draw(ctx) {
    ctx.beginPath()
    ctx.arc(this.pos.x, this.pos.y, this.size, 0, Math.PI * 2)
    ctx.fillStyle = this.color
    ctx.fill()
  }
}

class Particle {
  constructor(x, y, color) {
    this.pos = new Vector(x, y)
    this.vel = new Vector((Math.random() - 0.5) * 6, (Math.random() - 0.5) * 6)
    this.life = 20 + Math.random() * 10
    this.maxLife = this.life
    this.color = color
  }
  update() {
    this.pos.add(this.vel)
    this.life--
  }
  draw(ctx) {
    ctx.globalAlpha = Math.max(0, this.life / this.maxLife)
    ctx.fillStyle = this.color
    ctx.beginPath()
    ctx.arc(this.pos.x, this.pos.y, 1.5, 0, Math.PI * 2)
    ctx.fill()
    ctx.globalAlpha = 1.0
  }
}

class Entity {
  constructor(x, y, society, tribeId) {
    this.pos = new Vector(x, y)
    this.vel = new Vector((Math.random() - 0.5) * 2, (Math.random() - 0.5) * 2)
    this.vel.normalize()
    this.vel.mult(society.speed)
    this.society = society
    
    this.gender = Math.random() < 0.5 ? 'male' : 'female'
    this.tribeId = tribeId

    // Stats
    this.hp = CONFIG.maxHp
    this.cooldown = 0
    this.kills = 0
    this.foodEaten = 0
    this.age = 0
    this.personalMaxAge = CONFIG.maxAge + (Math.random() * 1200 - 600) // Random variance so they don't all die at once
    this.title = 'peasant'
    this.diseased = false
    
    this.memory = {}
    
    this.size = 3
    this.dmgMult = 1

    // RL fields — only used when rlMode is on
    this._rlMoveState = null
    this._rlMoveAction = -1
    this._rlPrevHp = this.hp
    this._rlFrameReward = 0
    this._rlCombatState = null
    this._rlCombatAction = -1
  }

  isChild() {
    return this.age < CONFIG.adultAge
  }

  canFight() {
    if (this.isChild()) return false
    if (this.gender === 'female' && !this.society.allowWomenFighters) return false
    return true
  }

  updateTitle() {
    if (this.title === 'king') {
      this.size = 8
      this.dmgMult = 1.5
      return
    }
    if (this.kills >= 3) {
      this.title = 'commander'
      this.size = 5
      this.dmgMult = 2.5 
    } else if (this.foodEaten >= 10) {
      this.title = 'minister'
      this.size = 4
      this.dmgMult = 1.2
    } else {
      this.title = 'peasant'
      this.size = 3
      this.dmgMult = 1
    }
  }

  update() {
    this.age++
    this.updateTitle()

    // Old age mechanic
    if (this.age > this.personalMaxAge) {
      this.hp = 0
      logEvent(`A ${this.title} of ${this.society.name} died of old age.`, '#888')
    }

    // Disease mechanic
    if (this.diseased) {
      this.hp -= 0.1 // Reduced disease severity
    }

    // Season damage (Winter)
    this.hp -= SEASONS[currentSeasonIdx].coldDmg

    // Movement: RL brain (if enabled) or classic heuristic
    if (rlMode && this.society.brain) {
      const state = getEntityState(this)
      // Store previous experience before overwriting state
      if (this._rlMoveState !== null) {
        const reward = (this.hp - this._rlPrevHp) * 0.5 + this._rlFrameReward
        this.society.brain.rememberMove(this._rlMoveState, this._rlMoveAction, reward, state)
      }
      this._rlFrameReward = 0
      const action = this.society.brain.chooseMove(state)
      this._rlMoveState = state
      this._rlMoveAction = action
      this._rlPrevHp = this.hp
      // Apply chosen direction directly — the network steers, speed applied below
      const dir = MOVE_DIRS[action]
      this.vel.x = dir[0]
      this.vel.y = dir[1]
    } else {
      // Classic heuristic AI
      let desired = null
      let recordDist = 800

      if (this.hp < 70) {
        for (let i = 0; i < simulation.foods.length; i++) {
          const f = simulation.foods[i]
          const d = this.pos.sub(f.pos).mag()
          if (d < recordDist) { recordDist = d; desired = f.pos.sub(this.pos) }
        }
      } else {
        let foundMate = false
        for (let i = 0; i < simulation.entities.length; i++) {
          const other = simulation.entities[i]
          if (other !== this && other.society === this.society && other.hp > 70 && other.gender !== this.gender) {
            const d = this.pos.sub(other.pos).mag()
            if (d < recordDist) { recordDist = d; desired = other.pos.sub(this.pos); foundMate = true }
          }
        }
        if (!foundMate) {
          for (let i = 0; i < simulation.entities.length; i++) {
            const other = simulation.entities[i]
            if (other !== this && other.society === this.society) {
              const d = this.pos.sub(other.pos).mag()
              if (d < recordDist) { recordDist = d; desired = other.pos.sub(this.pos) }
            }
          }
        }
      }

      if (desired) {
        desired.normalize()
        desired.mult(0.5)
        this.vel.add(desired)
      } else {
        const targetVec = new Vector((Math.random() - 0.5) * 0.5, (Math.random() - 0.5) * 0.5)
        const cos = Math.cos(targetVec.x)
        const sin = Math.sin(targetVec.y)
        const vx = this.vel.x * cos - this.vel.y * sin
        const vy = this.vel.x * sin + this.vel.y * cos
        this.vel.x = vx
        this.vel.y = vy
      }
    }
    
    const gridX = Math.floor(this.pos.x / CONFIG.gridSize)
    const gridY = Math.floor(this.pos.y / CONFIG.gridSize)
    
    if (gridX >= 0 && gridX < simulation.cols && gridY >= 0 && gridY < simulation.rows) {
      const cellOwner = simulation.territory[gridX][gridY]
      if (cellOwner !== this.society && cellOwner !== null) {
        if (simulation.isProtectedInterior(gridX, gridY)) {
          // Deep civilian territory is protected; intruders are pushed back toward the border.
          this.vel.mult(-1.4)
        } else {
          // Enemy border territory: home-field advantage for them, limited capture for combatants.
          this.hp -= 0.02
          if (this.canFight() && Math.random() < 0.03) simulation.territory[gridX][gridY] = this.society
        }
      } else if (cellOwner === null) {
        // Neutral land
        if (this.canFight() && Math.random() < 0.08) simulation.territory[gridX][gridY] = this.society
      } else {
        // Our land: heal
        if (this.hp < CONFIG.maxHp && !this.diseased) this.hp += 0.04
      }
    }

    let currentSpeed = this.society.speed
    if (this.title === 'minister') currentSpeed *= 1.3
    if (this.diseased) currentSpeed *= 0.5 
    
    // Normalize and scale velocity to exact speed so they don't accelerate infinitely
    this.vel.normalize()
    this.vel.mult(currentSpeed)

    this.pos.add(this.vel)

    if (this.pos.x < 0 || this.pos.x > width) this.vel.x *= -1
    if (this.pos.y < 0 || this.pos.y > height) this.vel.y *= -1
    this.pos.x = Math.max(0, Math.min(width, this.pos.x))
    this.pos.y = Math.max(0, Math.min(height, this.pos.y))

    if (this.cooldown > 0) this.cooldown--
    this.hp -= 0.03 // Reduced passive drain
  }

  draw(ctx) {
    ctx.beginPath()
    ctx.arc(this.pos.x, this.pos.y, this.size, 0, Math.PI * 2)
    ctx.fillStyle = this.diseased ? '#9acd32' : this.society.color // Yellow-green if diseased
    ctx.globalAlpha = Math.max(0.2, this.hp / CONFIG.maxHp)
    ctx.fill()
    ctx.globalAlpha = 1.0

    if (this.title === 'king') {
      ctx.beginPath()
      ctx.arc(this.pos.x, this.pos.y, this.size + 4, 0, Math.PI * 2)
      ctx.strokeStyle = '#ffd700' 
      ctx.lineWidth = 2
      ctx.stroke()
    } else if (this.title === 'commander') {
      ctx.beginPath()
      ctx.arc(this.pos.x, this.pos.y, this.size + 2, 0, Math.PI * 2)
      ctx.strokeStyle = '#ff4444'
      ctx.lineWidth = 1
      ctx.stroke()
    } else if (this.title === 'minister') {
      ctx.beginPath()
      ctx.arc(this.pos.x, this.pos.y, this.size + 2, 0, Math.PI * 2)
      ctx.strokeStyle = '#00ff88'
      ctx.lineWidth = 1
      ctx.stroke()
    }
  }

  decideAction(opponent) {
    // RL combat decision — brain outputs 0=hawk or 1=dove
    if (rlMode && this.society.brain) {
      const state = getEntityState(this)
      const action = this.society.brain.chooseCombat(state)
      this._rlCombatState = state
      this._rlCombatAction = action
      return action === 0 ? 'hawk' : 'dove'
    }

    const oppId = opponent.society.id
    if (!this.memory[oppId]) this.memory[oppId] = []

    const mem = this.memory[oppId]
    const strat = this.society.strategy

    if (strat === 'aggressive') return 'hawk'
    if (strat === 'defensive') return 'dove'
    
    if (strat === 'cooperative') {
      if (mem.length === 0) return 'dove'
      return mem[mem.length - 1] 
    }
    
    if (strat === 'grudger') {
      if (mem.includes('hawk')) return 'hawk'
      return 'dove'
    }
    
    if (strat === 'detective') {
      if (mem.length === 0) return 'dove'
      if (mem.length === 1) return 'hawk'
      if (mem.length === 2) return 'dove'
      if (mem.length === 3) return 'dove'
      
      const opponentCheated = mem.includes('hawk')
      if (opponentCheated) return mem[mem.length - 1] 
      else return 'hawk' 
    }

    return 'dove' 
  }

  interact(other) {
    if (this.cooldown > 0 || other.cooldown > 0) return

    // Disease spread: drastically lowered chance to prevent instant society wipe
    if (this.diseased && !other.diseased && Math.random() < 0.01) other.diseased = true
    if (!this.diseased && other.diseased && Math.random() < 0.01) this.diseased = true

    if (this.society === other.society) {
      if (this.gender !== other.gender) {
        // Lowered threshold to 70 for reproduction
        if (this.hp > 70 && other.hp > 70 && Math.random() < 0.05) {
          this.hp -= CONFIG.reproductionCost / 2
          other.hp -= CONFIG.reproductionCost / 2
          
          // Inherit the mother's tribe
          const inheritedTribe = this.gender === 'female' ? this.tribeId : other.tribeId
          simulation.spawnChild(this.pos.x, this.pos.y, this.society, inheritedTribe)
        }
      }
      return
    }

    if (!this.canFight() || !other.canFight() || !simulation.isCombatZone(this, other)) {
      this.vel.mult(-1)
      other.vel.mult(-1)
      return
    }

    this.cooldown = 30
    other.cooldown = 30

    const thisHpBefore = this.hp
    const otherHpBefore = other.hp

    let actA = this.decideAction(other)
    let actB = other.decideAction(this)

    const myMem = this.memory[other.society.id]
    const oppMem = other.memory[this.society.id]
    myMem.push(actB)
    oppMem.push(actA)

    const dmgA = CONFIG.baseDamage * this.dmgMult
    const dmgB = CONFIG.baseDamage * other.dmgMult

    let damageDealt = false

    if (actA === 'hawk' && actB === 'hawk') {
      this.hp -= dmgB * 1.5; other.hp -= dmgA * 1.5
      damageDealt = true
    } else if (actA === 'hawk' && actB === 'dove') {
      this.hp += 10; other.hp -= dmgA
      damageDealt = true
    } else if (actA === 'dove' && actB === 'hawk') {
      this.hp -= dmgB; other.hp += 10
      damageDealt = true
    } else if (actA === 'dove' && actB === 'dove') {
      this.vel.mult(-1); other.vel.mult(-1)
    } 

    if (damageDealt) {
      // Spawn combat sparks
      for(let i=0; i<4; i++) {
        simulation.particles.push(new Particle(this.pos.x, this.pos.y, '#ff4444'))
      }
    }

    if (this.hp <= 0 && other.hp > 0) {
      other.kills++
      if (other.kills === 3) logEvent(`A peasant of ${other.society.name} became a Commander!`, other.society.color)
    }
    if (other.hp <= 0 && this.hp > 0) {
      this.kills++
      if (this.kills === 3) logEvent(`A peasant of ${this.society.name} became a Commander!`, this.society.color)
    }

    // Record RL combat experiences — reward = HP delta + kill bonus
    if (rlMode) {
      if (this._rlCombatState !== null) {
        const nextState = getEntityState(this)
        const killBonus = (this.hp > 0 && other.hp <= 0) ? 2.0 : 0
        this.society.brain.rememberCombat(
          this._rlCombatState, this._rlCombatAction,
          (this.hp - thisHpBefore) * 0.3 + killBonus,
          nextState
        )
        this._rlCombatState = null
      }
      if (other._rlCombatState !== null) {
        const nextState = getEntityState(other)
        const killBonus = (other.hp > 0 && this.hp <= 0) ? 2.0 : 0
        other.society.brain.rememberCombat(
          other._rlCombatState, other._rlCombatAction,
          (other.hp - otherHpBefore) * 0.3 + killBonus,
          nextState
        )
        other._rlCombatState = null
      }
    }
  }
}

// --- THIRD-PARTY LIBRARIES ---

// Quadtree (timohausmann/quadtree-js — MIT)
// Reduces entity-entity collision from O(n²) to O(n log n)
function Quadtree(bounds, max_objects, max_levels, level) {
  this.max_objects = max_objects || 10
  this.max_levels  = max_levels  || 4
  this.level       = level       || 0
  this.bounds      = bounds
  this.objects     = []
  this.nodes       = []
}
Quadtree.prototype.split = function() {
  const nL = this.level + 1, sw = this.bounds.width / 2, sh = this.bounds.height / 2
  const x = this.bounds.x, y = this.bounds.y
  this.nodes[0] = new Quadtree({ x: x + sw, y,        width: sw, height: sh }, this.max_objects, this.max_levels, nL)
  this.nodes[1] = new Quadtree({ x,         y,        width: sw, height: sh }, this.max_objects, this.max_levels, nL)
  this.nodes[2] = new Quadtree({ x,         y: y + sh, width: sw, height: sh }, this.max_objects, this.max_levels, nL)
  this.nodes[3] = new Quadtree({ x: x + sw, y: y + sh, width: sw, height: sh }, this.max_objects, this.max_levels, nL)
}
Quadtree.prototype.getIndex = function(r) {
  const idx = [], vm = this.bounds.x + this.bounds.width / 2, hm = this.bounds.y + this.bounds.height / 2
  const n = r.y < hm, s = r.y + r.height > hm, w = r.x < vm, e = r.x + r.width > vm
  if (n && e) idx.push(0)
  if (n && w) idx.push(1)
  if (s && w) idx.push(2)
  if (s && e) idx.push(3)
  return idx
}
Quadtree.prototype.insert = function(r) {
  if (this.nodes.length) { this.getIndex(r).forEach(i => this.nodes[i].insert(r)); return }
  this.objects.push(r)
  if (this.objects.length > this.max_objects && this.level < this.max_levels) {
    if (!this.nodes.length) this.split()
    this.objects.forEach(o => this.getIndex(o).forEach(i => this.nodes[i].insert(o)))
    this.objects = []
  }
}
Quadtree.prototype.retrieve = function(r) {
  let ret = [...this.objects]
  if (this.nodes.length) this.getIndex(r).forEach(i => ret = ret.concat(this.nodes[i].retrieve(r)))
  return this.level === 0 ? [...new Set(ret)] : ret
}
Quadtree.prototype.clear = function() {
  this.objects = []
  this.nodes.forEach(n => n.clear())
  this.nodes = []
}

// Fantasy name generator (skeeto/fantasyname — Public Domain)
// Used for procedurally named kings and queens
String.prototype.combinations = function() { return 1 }
String.prototype.min = function() { return this.length }
String.prototype.max = function() { return this.length }
String.prototype.enumerate = function() { return [String(this)] }
var NameGen = {}
NameGen.symbols = {
  s: ['ach','ack','ad','age','ald','ale','an','ang','ar','ard','as','ash','at','ath','augh','aw','ban','bel','bur','cer','cha','che','dan','dar','del','den','dra','dyn','eld','elm','em','en','end','eng','enth','er','ess','est','et','gar','hat','hin','hon','ia','ight','ild','im','ina','ine','ing','ir','is','iss','it','kal','kel','kim','kin','ler','lor','lye','mor','mos','nal','ny','old','om','on','or','orm','os','ough','per','pol','qua','que','rad','rak','ran','ray','ril','ris','rod','roth','ryn','sam','say','ser','shy','skel','sul','tai','tan','tas','ther','tia','tin','ton','tor','tur','um','und','unt','urn','usk','ust','ver','ves','vor','war','wor','yer'],
  v: ['a','e','i','o','u','y'],
  V: ['a','e','i','o','u','y','ae','ai','au','ay','ea','ee','ei','eu','ey','ia','ie','oe','oi','oo','ou','ui'],
  c: ['b','c','d','f','g','h','j','k','l','m','n','p','q','r','s','t','v','w','x','y','z'],
  B: ['b','bl','br','c','ch','chr','cl','cr','d','dr','f','g','h','j','k','l','ll','m','n','p','ph','qu','r','rh','s','sch','sh','sl','sm','sn','st','str','sw','t','th','thr','tr','v','w','wh','y','z','zh'],
  C: ['b','c','ch','ck','d','f','g','gh','h','k','l','ld','ll','lt','m','n','nd','nn','nt','p','ph','q','r','rd','rr','rt','s','sh','ss','st','t','th','v','w','y','z'],
}
NameGen._isString = o => Object.prototype.toString.call(o) === '[object String]'
NameGen._compress = function(a) {
  const emit = [], accum = []
  const dump = () => { if (accum.length) { emit.push(accum.join('')); accum.length = 0 } }
  a.forEach(x => NameGen._isString(x) ? accum.push(x) : (dump(), emit.push(x)))
  dump(); return emit
}
NameGen._capitalize = s => s.replace(/^./, c => c.toUpperCase())
NameGen._reverse    = s => s.split(/(?:)/).reverse().join('')
NameGen.Random = function Random(gs) {
  if (!(this instanceof NameGen.Random)) {
    return gs.length === 0 ? '' : gs.length === 1 ? gs[0] : new NameGen.Random(gs)
  }
  this.sub = gs
}
NameGen.Random.prototype.toString = function() { return this.sub.length ? this.sub[Math.floor(Math.random() * this.sub.length)].toString() : '' }
NameGen.Random.prototype.combinations = function() { return Math.max(1, this.sub.reduce((t, g) => t + g.combinations(), 0)) }
NameGen.Random.prototype.min = function() { return Math.min(...this.sub.map(g => g.min())) }
NameGen.Random.prototype.max = function() { return Math.max(...this.sub.map(g => g.max())) }
NameGen.Random.prototype.enumerate = function() { return [].concat(...this.sub.map(g => g.enumerate())) }
NameGen.Sequence = function Sequence(gs) {
  gs = NameGen._compress(gs)
  if (!(this instanceof NameGen.Sequence)) {
    return gs.length === 0 ? '' : gs.length === 1 ? gs[0] : new NameGen.Sequence(gs)
  }
  this.sub = gs
}
NameGen.Sequence.prototype.toString = function() { return this.sub.join('') }
NameGen.Sequence.prototype.combinations = function() { return this.sub.reduce((t, g) => t * g.combinations(), 1) }
NameGen.Sequence.prototype.min = function() { return this.sub.reduce((t, g) => t + g.min(), 0) }
NameGen.Sequence.prototype.max = function() { return this.sub.reduce((t, g) => t + g.max(), 0) }
NameGen.Sequence.prototype.enumerate = function() {
  const enums = this.sub.map(g => g.enumerate())
  const enumerate = (enums, prefix) => enums.length === 1 ? enums[0].map(e => prefix + e)
    : [].concat(...enums[0].map((_, i) => enumerate(enums.slice(1), prefix + enums[0][i])))
  return enumerate(enums, '')
}
NameGen.fromTransform = function(f) {
  function G(g) {
    if (!(this instanceof G)) return NameGen._isString(g) ? f(g) : new G(g)
    this.generator = g
  }
  G.prototype.toString = function() { return f(this.generator.toString()) }
  G.prototype.combinations = function() { return this.generator.combinations() }
  G.prototype.min = function() { return this.generator.min() }
  G.prototype.max = function() { return this.generator.max() }
  G.prototype.enumerate = function() { return this.generator.enumerate().map(f) }
  return G
}
NameGen.Capitalizer = NameGen.fromTransform(NameGen._capitalize)
NameGen.Reverser    = NameGen.fromTransform(NameGen._reverse)
NameGen._Group = function() { this.set = [[]]; this.wrappers = [] }
NameGen._Group.prototype.add = function(g) {
  while (this.wrappers.length) { const t = this.wrappers.pop(); g = t(g) }
  this.set[this.set.length - 1].push(g); return this
}
NameGen._Group.prototype.split = function() { this.set.push([]); return this }
NameGen._Group.prototype.wrap  = function(t) { this.wrappers.push(t); return this }
NameGen._Group.prototype.emit  = function() { return NameGen.Random(this.set.map(NameGen.Sequence)) }
NameGen._Literal = function() { NameGen._Group.call(this) }
NameGen._Literal.prototype = Object.create(NameGen._Group.prototype)
NameGen._Symbol  = function() { NameGen._Group.call(this) }
NameGen._Symbol.prototype = Object.create(NameGen._Group.prototype)
NameGen._Symbol.prototype.add = function(g, literal) {
  if (!literal) g = NameGen.Random(NameGen.symbols[g] || [g])
  NameGen._Group.prototype.add.call(this, g); return this
}
NameGen.compile = function(input) {
  const stack = []; stack.top = () => stack[stack.length - 1]
  stack.push(new NameGen._Symbol())
  for (const c of input) {
    switch (c) {
      case '<': stack.push(new NameGen._Symbol()); break
      case '(': stack.push(new NameGen._Literal()); break
      case '>': case ')': { const last = stack.pop().emit(); stack.top().add(last, true); break }
      case '|': stack.top().split(); break
      case '!': stack.top() instanceof NameGen._Symbol ? stack.top().wrap(NameGen.Capitalizer) : stack.top().add(c); break
      case '~': stack.top() instanceof NameGen._Symbol ? stack.top().wrap(NameGen.Reverser) : stack.top().add(c); break
      default:  stack.top().add(c)
    }
  }
  return stack.top().emit()
}
// Pre-compiled king/queen name generator
const kingNameGen = NameGen.compile('!Bs|!BsV|!BVs|!BsCv')

// Territory border tracer (scottglz/marching-squares — open source)
// Traces a closed polygon around each society's territory region
const _MS_UP=[0,-1,1],_MS_DOWN=[0,1,0],_MS_LEFT=[-1,0,1],_MS_RIGHT=[1,0,0]
const _MS_TRANS=[null,[_MS_LEFT,_MS_LEFT],[_MS_UP,_MS_UP],[_MS_LEFT,_MS_LEFT],[_MS_DOWN,_MS_DOWN],[_MS_DOWN,_MS_DOWN],[_MS_UP,_MS_DOWN],[_MS_DOWN,_MS_DOWN],[_MS_RIGHT,_MS_RIGHT],[_MS_RIGHT,_MS_LEFT],[_MS_UP,_MS_UP],[_MS_LEFT,_MS_LEFT],[_MS_RIGHT,_MS_RIGHT],[_MS_RIGHT,_MS_RIGHT],[_MS_UP,_MS_UP]]
function traceRegion(x, y, isInside) {
  const startX = x, startY = y
  const ret = [{x, y}]
  let dir = _MS_DOWN
  let square = (isInside(x-1,y-1)?1:0)+(isInside(x,y-1)?2:0)+(isInside(x-1,y)?4:0)+(isInside(x,y)?8:0)
  if (square===0||square===15) throw new Error('Bad start')
  while (true) {
    dir = _MS_TRANS[square][dir[2]]
    x += dir[0]; y += dir[1]
    if (x===startX && y===startY) return ret
    ret.push({x,y})
    if      (dir===_MS_DOWN)  square=((square&12)>>2)
    else if (dir===_MS_UP)    square=((square&3)<<2)
    else if (dir===_MS_RIGHT) square=((square&10)>>1)
    else if (dir===_MS_LEFT)  square=((square&5)<<1)
    if      (dir===_MS_DOWN||dir===_MS_LEFT)  square+=(isInside(x-1,y)?4:0)
    else                                       square+=(isInside(x,y-1)?2:0)
    if      (dir===_MS_DOWN||dir===_MS_RIGHT) square+=(isInside(x,y)?8:0)
    else                                       square+=(isInside(x-1,y-1)?1:0)
  }
}

// Draw crisp territory outlines via marching squares polygon tracing
function drawTerritoryBorders(targetCtx) {
  const gs = CONFIG.gridSize
  for (const soc of simulation.societies) {
    const isInside = (x, y) => simulation.isInsideGrid(x, y) && simulation.territory[x][y] === soc
    const visitedCorners = new Set()
    for (let gx = 1; gx <= simulation.cols; gx++) {
      for (let gy = 1; gy <= simulation.rows; gy++) {
        if (visitedCorners.has(gx * 10000 + gy)) continue
        const bits = (isInside(gx-1,gy-1)?1:0)+(isInside(gx,gy-1)?2:0)+(isInside(gx-1,gy)?4:0)+(isInside(gx,gy)?8:0)
        if (bits===0||bits===15) continue
        try {
          const polygon = traceRegion(gx, gy, isInside)
          if (polygon.length < 3) continue
          polygon.forEach(p => visitedCorners.add(p.x * 10000 + p.y))
          targetCtx.beginPath()
          targetCtx.moveTo(polygon[0].x * gs, polygon[0].y * gs)
          for (let i = 1; i < polygon.length; i++) targetCtx.lineTo(polygon[i].x * gs, polygon[i].y * gs)
          targetCtx.closePath()
          targetCtx.strokeStyle = soc.color
          targetCtx.lineWidth = 1.5
          targetCtx.globalAlpha = 0.55
          targetCtx.stroke()
        } catch(e) { /* bad starting point — skip */ }
      }
    }
  }
  targetCtx.globalAlpha = 1.0
}

// --- RL STATE OBSERVATION ---

function getEntityState(entity) {
  const scanR = 350 // vision radius — limits O(n²) cost
  let fdx = 0, fdy = 0, fdist = 1
  let minFd = scanR
  for (const f of simulation.foods) {
    const dx = f.pos.x - entity.pos.x, dy = f.pos.y - entity.pos.y
    if (Math.abs(dx) > scanR || Math.abs(dy) > scanR) continue
    const d = Math.sqrt(dx * dx + dy * dy)
    if (d < minFd) {
      minFd = d; fdist = d / scanR
      fdx = dx / (d + 1e-6); fdy = dy / (d + 1e-6)
    }
  }
  let edx = 0, edy = 0, edist = 1, ehp = 1
  let minEd = scanR
  for (const e of simulation.entities) {
    if (e.society === entity.society) continue
    const dx = e.pos.x - entity.pos.x, dy = e.pos.y - entity.pos.y
    if (Math.abs(dx) > scanR || Math.abs(dy) > scanR) continue
    const d = Math.sqrt(dx * dx + dy * dy)
    if (d < minEd) {
      minEd = d; edist = d / scanR
      edx = dx / (d + 1e-6); edy = dy / (d + 1e-6)
      ehp = e.hp / CONFIG.maxHp
    }
  }
  let frdist = 1, minFrd = scanR
  for (const e of simulation.entities) {
    if (e === entity || e.society !== entity.society) continue
    const dx = e.pos.x - entity.pos.x, dy = e.pos.y - entity.pos.y
    if (Math.abs(dx) > scanR || Math.abs(dy) > scanR) continue
    const d = Math.sqrt(dx * dx + dy * dy)
    if (d < minFrd) { minFrd = d; frdist = d / scanR }
  }
  const gx = Math.floor(entity.pos.x / CONFIG.gridSize)
  const gy = Math.floor(entity.pos.y / CONFIG.gridSize)
  const onOwn = simulation.isInsideGrid(gx, gy) && simulation.territory[gx][gy] === entity.society ? 1 : 0
  return new Float32Array([
    entity.hp / CONFIG.maxHp,                        // 0: own HP
    Math.min(1, entity.age / entity.personalMaxAge), // 1: age
    fdist, fdx, fdy,                                  // 2-4: food vector
    edist, edx, edy,                                  // 5-7: nearest enemy vector
    ehp,                                              // 8: enemy HP
    onOwn,                                            // 9: on own territory
    frdist,                                           // 10: nearest friendly dist
  ])
}

// --- ENGINE ---

const simulation = {
  societies: [],
  entities: [],
  foods: [],
  particles: [],
  territory: [],
  cols: 0,
  rows: 0,
  nextSocietyId: 3,
  nextTribeId: 1,
  populationHistory: [], 

  init(soc1Config, soc2Config) {
    historyLog.innerHTML = '' // Clear log
    historyLogLarge.innerHTML = ''
    logEvent('The simulation has begun.', '#fff')

    this.societies = [
      new Society(1, 'Alpha', '#00f0ff', soc1Config.strat, soc1Config.agg, soc1Config.spd, soc1Config.allowWomenFighters),
      new Society(2, 'Beta', '#ff0055', soc2Config.strat, soc2Config.agg, soc2Config.spd, soc2Config.allowWomenFighters)
    ]
    logEvent('Founding combat rules are locked and inherited by descendants.', '#ffd166')
    if (rlMode) {
      this.societies.forEach(soc => { soc.brain = new RLBrain() })
      logEvent('RL Mode: each society runs a shared neural Q-network. Exploration begins at ε=1.0.', '#a78bfa')
    }
    this.entities = []
    this.foods = []
    this.particles = []
    this.populationHistory = []
    
    this.cols = Math.ceil(width / CONFIG.gridSize)
    this.rows = Math.ceil(height / CONFIG.gridSize)
    this.territory = new Array(this.cols).fill(0).map(() => new Array(this.rows).fill(null))

    const cornerSize = 8 
    for(let x = this.cols - cornerSize; x < this.cols; x++) {
      for(let y = 0; y < cornerSize; y++) {
        if (x >= 0 && y >= 0) this.territory[x][y] = this.societies[0]
      }
    }
    
    for(let x = 0; x < cornerSize; x++) {
      for(let y = this.rows - cornerSize; y < this.rows; y++) {
        if (x < this.cols && y < this.rows) this.territory[x][y] = this.societies[1]
      }
    }

    this.nextTribeId = 1
    for(let i=0; i<CONFIG.startPopulation; i++) {
      const aX = width - (Math.random() * (cornerSize * CONFIG.gridSize))
      const aY = Math.random() * (cornerSize * CONFIG.gridSize)
      const alphaAdult = new Entity(aX, aY, this.societies[0], this.nextTribeId++)
      alphaAdult.age = CONFIG.adultAge + Math.random() * CONFIG.adultAge
      this.entities.push(alphaAdult)

      const bX = Math.random() * (cornerSize * CONFIG.gridSize)
      const bY = height - (Math.random() * (cornerSize * CONFIG.gridSize))
      const betaAdult = new Entity(bX, bY, this.societies[1], this.nextTribeId++)
      betaAdult.age = CONFIG.adultAge + Math.random() * CONFIG.adultAge
      this.entities.push(betaAdult)
    }
    
    this.recordHistory()
    this.updateSeasonUI()
  },

  spawnChild(x, y, society, tribeId) {
    const child = new Entity(x, y, society, tribeId)
    child.hp = 40
    this.entities.push(child)
  },

  triggerTribeRebellion(leader) {
    const targetTribe = leader.tribeId
    const oldSociety = leader.society

    const newColor = getRandomColor()
    const strats = ['aggressive', 'defensive', 'cooperative', 'grudger', 'detective']
    const newSoc = new Society(
      this.nextSocietyId++, 
      `Cult of Tribe-${targetTribe}`, 
      newColor, 
      strats[Math.floor(Math.random() * strats.length)], 
      Math.random() * 100,
      oldSociety.speed * 1.1,
      oldSociety.allowWomenFighters
    )
    
    if (rlMode) newSoc.brain = new RLBrain(oldSociety.brain) // inherit + mutate
    this.societies.push(newSoc)

    logEvent(`REVOLUTION! A ${leader.title} led their Tribe's bloodline away from ${oldSociety.name} to form a Cult!`, newColor)

    // Convert the entire bloodline tribe
    this.entities.forEach(e => {
      if (e.tribeId === targetTribe && e.society === oldSociety) {
        e.society = newSoc
        // Reset ranks so they don't immediately chain rebel
        e.kills = 0
        e.foodEaten = 0
        e.title = 'peasant'
      }
    })

    leader.title = 'king'
  },

  updateSeasonUI() {
    const s = SEASONS[currentSeasonIdx]
    seasonDisplay.innerText = s.name
    seasonDisplay.style.color = s.color
  },

  recordHistory() {
    const popMap = {}
    this.societies.forEach(s => popMap[s.id] = { pop: 0, color: s.color })
    this.entities.forEach(e => popMap[e.society.id].pop++)
    
    this.populationHistory.push({
      year: currentYear,
      data: popMap
    })
    
    this.drawGraph()
  },

  getGridAt(pos) {
    return {
      x: Math.floor(pos.x / CONFIG.gridSize),
      y: Math.floor(pos.y / CONFIG.gridSize)
    }
  },

  isInsideGrid(x, y) {
    return x >= 0 && x < this.cols && y >= 0 && y < this.rows
  },

  isBorderCell(x, y) {
    if (!this.isInsideGrid(x, y)) return false
    const owner = this.territory[x][y]
    if (!owner) return true

    const neighbors = [
      { x: x - 1, y },
      { x: x + 1, y },
      { x, y: y - 1 },
      { x, y: y + 1 }
    ]

    return neighbors.some(cell => this.isInsideGrid(cell.x, cell.y) && this.territory[cell.x][cell.y] !== owner)
  },

  isProtectedInterior(x, y) {
    return this.isInsideGrid(x, y) && this.territory[x][y] !== null && !this.isBorderCell(x, y)
  },

  isCombatZone(entityA, entityB) {
    const midGrid = this.getGridAt({
      x: (entityA.pos.x + entityB.pos.x) / 2,
      y: (entityA.pos.y + entityB.pos.y) / 2
    })

    return this.isBorderCell(midGrid.x, midGrid.y) && !this.isProtectedInterior(midGrid.x, midGrid.y)
  },

  drawGraph() {
    drawPopulationGraph(graphCanvas, gCtx)
    drawPopulationGraph(graphCanvasLarge, gLargeCtx)
  },

  update() {
    // Build spatial index for entity-entity collision (O(n log n) vs O(n²))
    const qt = new Quadtree({ x: 0, y: 0, width, height }, 10, 5)
    for (const e of this.entities) {
      qt.insert({ x: e.pos.x - 16, y: e.pos.y - 16, width: 32, height: 32, _e: e })
    }
    this._qt = qt // expose for getEntityState RL scans

    // Patient Zero Plague Spawning
    if (this.entities.length > 50 && Math.random() < 0.0005) {
      const luckyOne = this.entities[Math.floor(Math.random() * this.entities.length)]
      if (!luckyOne.diseased) {
        luckyOne.diseased = true
        logEvent(`A plague has started in ${luckyOne.society.name}!`, '#9acd32')
      }
    }

    this.societies.forEach(soc => {
      let bestScore = -1
      let newKing = null
      
      const socEntities = this.entities.filter(e => e.society === soc)
      if (socEntities.length === 0) {
        // Extinction log
        if (soc.king) {
          logEvent(`${soc.name} has been wiped out!`, '#ff0000')
          soc.king = null
        }
        return
      }

      socEntities.forEach(e => {
        if (e.title !== 'king') { 
            if (e.title !== 'commander' && e.title !== 'minister') e.title = 'peasant'
        }
        const score = e.kills * 5 + e.foodEaten * 2 + e.age * 0.01
        if (score > bestScore) {
          bestScore = score
          newKing = e
        }
      })
      
      if (newKing) {
        if (soc.king && soc.king !== newKing) soc.king.title = 'peasant'
        if (soc.king !== newKing) {
          if (!newKing._rulerName) newKing._rulerName = kingNameGen.toString()
          const title = newKing.gender === 'female' ? 'Queen' : 'King'
          logEvent(`${soc.name} has crowned ${title} ${newKing._rulerName}!`, soc.color)
        }
        newKing.title = 'king'
        soc.king = newKing
      }
    })

    // Count tribe sizes
    const tribeCounts = {}
    this.entities.forEach(e => {
      tribeCounts[e.tribeId] = (tribeCounts[e.tribeId] || 0) + 1
    })

    // Trigger Ideological Tribe Rebellions
    this.entities.forEach(e => {
      if ((e.title === 'commander' || e.title === 'minister') && tribeCounts[e.tribeId] >= 12 && Math.random() < 0.005) {
        this.triggerTribeRebellion(e)
      }
    })

    const activeFoodMult = SEASONS[currentSeasonIdx].foodMult

    if (activeFoodMult > 0 && this.foods.length < CONFIG.maxFood) {
      const landMap = new Map()
      const neutralCells = []
      
      for (let x = 0; x < this.cols; x++) {
        for (let y = 0; y < this.rows; y++) {
          const owner = this.territory[x][y]
          if (owner) {
            if (!landMap.has(owner)) landMap.set(owner, [])
            landMap.get(owner).push({x, y})
          } else {
            neutralCells.push({x, y})
          }
        }
      }

      if (neutralCells.length > 0 && Math.random() < 0.05 * activeFoodMult) {
        const cell = neutralCells[Math.floor(Math.random() * neutralCells.length)]
        this.foods.push(new Food(cell.x * CONFIG.gridSize + Math.random() * CONFIG.gridSize, cell.y * CONFIG.gridSize + Math.random() * CONFIG.gridSize))
      }

      landMap.forEach((cells, society) => {
        const spawnChance = cells.length * 0.01 * activeFoodMult
        if (Math.random() < spawnChance) {
          const cell = cells[Math.floor(Math.random() * cells.length)]
          this.foods.push(new Food(cell.x * CONFIG.gridSize + Math.random() * CONFIG.gridSize, cell.y * CONFIG.gridSize + Math.random() * CONFIG.gridSize))
        }
      })
    }

    const _qtProcessed = new Set()
    for (let i = this.entities.length - 1; i >= 0; i--) {
      const e = this.entities[i]
      e.update()

      for (let j = this.foods.length - 1; j >= 0; j--) {
        const f = this.foods[j]
        const d = e.pos.sub(f.pos).mag()
        if (d < e.size + f.size) {
          e.hp = Math.min(CONFIG.maxHp, e.hp + CONFIG.foodEnergy)
          e.foodEaten++
          if (rlMode) e._rlFrameReward += 1.5 // positive reward for eating
          this.foods.splice(j, 1)
        }
      }

      // Entity-entity collision — quadtree narrows candidates, Set deduplicates pairs
      const searchR = 20
      const candidates = qt.retrieve({ x: e.pos.x - searchR, y: e.pos.y - searchR, width: searchR * 2, height: searchR * 2 })
      for (const c of candidates) {
        const other = c._e
        if (other === e || _qtProcessed.has(other)) continue
        const d = e.pos.sub(other.pos).mag()
        if (d < e.size + other.size) {
          e.interact(other)
          const overlap = (e.size + other.size) - d
          const dir = e.pos.sub(other.pos)
          dir.normalize()
          dir.mult(overlap / 2)
          e.pos.add(dir)
          other.pos.sub(dir)
        }
      }
      _qtProcessed.add(e)

      if (e.hp <= 0) {
        if (rlMode && e._rlMoveState !== null) {
          // Terminal experience: big negative reward for dying
          const termState = getEntityState(e)
          e.society.brain.rememberMove(e._rlMoveState, e._rlMoveAction, -2.0, termState)
        }
        this.entities.splice(i, 1)
        if (!e.diseased) this.foods.push(new Food(e.pos.x, e.pos.y))
      }
    }

    // Update particles
    for(let i = this.particles.length - 1; i >= 0; i--) {
      this.particles[i].update()
      if(this.particles[i].life <= 0) this.particles.splice(i, 1)
    }

    // RL training — run a batch every 30 frames for each society brain
    if (rlMode && frames % 30 === 0) {
      for (const soc of this.societies) {
        if (soc.brain) soc.brain.train()
      }
    }
  },

  draw() {
    ctx.clearRect(0, 0, width, height)
    
    // Day/Night and Season Overlay
    const totalTime = currentYear + (frames / FRAMES_PER_YEAR)
    // Night cycles every 2 years
    const nightFactor = Math.max(0, Math.sin(totalTime * Math.PI)) 
    
    ctx.save()
    
    // Territory
    ctx.filter = 'blur(25px)' 
    const territoryPulse = Math.sin(frames / 10) * 0.05 + 0.35 // Breathing effect
    for (let x = 0; x < this.cols; x++) {
      for (let y = 0; y < this.rows; y++) {
        const owner = this.territory[x][y]
        if (owner) {
          ctx.fillStyle = owner.color
          ctx.globalAlpha = territoryPulse
          ctx.beginPath()
          const cx = x * CONFIG.gridSize + CONFIG.gridSize / 2
          const cy = y * CONFIG.gridSize + CONFIG.gridSize / 2
          ctx.arc(cx, cy, CONFIG.gridSize * 1.2, 0, Math.PI * 2)
          ctx.fill()
        }
      }
    }
    ctx.restore()

    // Crisp territory borders via marching squares polygon tracing
    drawTerritoryBorders(ctx)

    // Day/Night filter over territory
    if (nightFactor > 0) {
      ctx.fillStyle = `rgba(5, 5, 15, ${nightFactor * 0.4})`
      ctx.fillRect(0, 0, width, height)
    }

    this.foods.forEach(f => f.draw(ctx))
    this.particles.forEach(p => p.draw(ctx))
    
    // Entities glow at night
    if (nightFactor > 0.5) {
      ctx.shadowBlur = 10
    }
    this.entities.forEach(e => {
      ctx.shadowColor = e.society.color
      e.draw(ctx)
    })
    ctx.shadowBlur = 0

    // Mirror to sidebar preview when in graph view
    if (activeView === 'graph' && simPreviewCanvas.width > 0 && simPreviewCanvas.height > 0) {
      simPreviewCtx.clearRect(0, 0, simPreviewCanvas.width, simPreviewCanvas.height)
      simPreviewCtx.drawImage(canvas, 0, 0, simPreviewCanvas.width, simPreviewCanvas.height)
    }
  },

  updateStats() {
    const popMap = {}
    const terrMap = {}
    const fighterMap = {}
    const civilianMap = {}
    let diseasedCount = 0
    
    this.societies.forEach(s => {
      popMap[s.id] = 0
      terrMap[s.id] = 0
      fighterMap[s.id] = 0
      civilianMap[s.id] = 0
    })
    this.entities.forEach(e => { 
      popMap[e.society.id]++ 
      if (e.canFight()) fighterMap[e.society.id]++
      else civilianMap[e.society.id]++
      if (e.diseased) diseasedCount++
    })
    
    for (let x = 0; x < this.cols; x++) {
      for (let y = 0; y < this.rows; y++) {
        const owner = this.territory[x][y]
        if (owner) terrMap[owner.id]++
      }
    }

    const rankedSocieties = [...this.societies].sort((a, b) => {
      const popDiff = popMap[b.id] - popMap[a.id]
      if (popDiff !== 0) return popDiff
      return terrMap[b.id] - terrMap[a.id]
    })

    let html = `
      <div class="stats-table">
        <div class="stats-header">
          <span>Society</span>
          <span>Pop</span>
          <span>Land</span>
          <span>Fight</span>
          <span>Civil</span>
        </div>
    `
    rankedSocieties.forEach((s, index) => {
      if (popMap[s.id] > 0) {
        html += `
          <div class="stats-row">
            <span class="soc-name" style="color: ${s.color}" title="${s.name}">#${index + 1} ${s.name}</span>
            <span>${popMap[s.id]}</span>
            <span>${terrMap[s.id]}</span>
            <span>${fighterMap[s.id]}</span>
            <span>${civilianMap[s.id]}</span>
          </div>
        `
      }
    })
    html += `</div>`

    html += `<div class="stats-totals">`
    html += `<span>Total: ${this.entities.length}</span>`
    html += `<span style="color: #00ff88;">Food: ${this.foods.length}</span>`
    if (diseasedCount > 0) {
      html += `<span style="color: #9acd32; font-weight: bold;">Plague: ${diseasedCount}</span>`
    }
    html += `</div>`

    if (rlMode) {
      html += `<div class="rl-panel"><div class="rl-panel-title">🧠 RL Training</div>`
      html += `<div class="rl-header"><span>Society</span><span>ε %</span><span>Avg R</span><span>Buf</span></div>`
      for (const s of this.societies) {
        if (popMap[s.id] > 0 && s.brain) {
          const eps = (s.brain.epsilon * 100).toFixed(0)
          const avgR = s.brain.avgReward
          const buf = s.brain.moveBuffer.length
          html += `<div class="rl-row">
            <span class="rl-soc-name" style="color:${s.color}" title="${s.name}">${s.name}</span>
            <span>${eps}%</span>
            <span>${avgR}</span>
            <span>${buf}</span>
          </div>`
        }
      }
      html += `</div>`
    }

    statsContent.innerHTML = html
  }
}

function loop() {
  if (!isRunning) return

  if (isPaused) {
    simulation.draw()
    animationId = requestAnimationFrame(loop)
    return
  }
  
  for (let s = 0; s < simSpeed; s++) {
    frames++
    
    // Seasons change 4 times per year (every FRAMES_PER_YEAR / 4 frames)
    let newSeasonIdx = Math.floor(frames / (FRAMES_PER_YEAR / 4))
    if (newSeasonIdx >= 4) newSeasonIdx = 3 // clamp just in case
    
    if (newSeasonIdx !== currentSeasonIdx) {
      currentSeasonIdx = newSeasonIdx
      simulation.updateSeasonUI()
    }

    if (frames >= FRAMES_PER_YEAR) {
      frames = 0
      currentYear++
      document.getElementById('year-display').innerText = `Year: ${currentYear} / ${targetYears}`
      
      // Happy New Year log
      if (currentYear % 10 === 0) logEvent(`Decade ${currentYear} begins.`, '#ffffff')

      simulation.recordHistory()

      if (currentYear >= targetYears) {
        isRunning = false
        isPaused = false
        pauseBtn.innerText = 'Pause'
        document.getElementById('year-display').innerText = `Simulation Finished (Year ${currentYear})`
        simulation.draw()
        simulation.updateStats()
        return
      }
    }
    simulation.update()
  }

  simulation.draw()
  
  if (Math.random() < 0.1) simulation.updateStats()

  animationId = requestAnimationFrame(loop)
}

// --- UI LOGIC ---

speedControl.addEventListener('input', (e) => {
  simSpeed = parseInt(e.target.value)
  speedLabel.innerText = simSpeed + 'x'
})

pauseBtn.addEventListener('click', () => {
  if (!isRunning) return
  isPaused = !isPaused
  pauseBtn.innerText = isPaused ? 'Resume' : 'Pause'
  if (isPaused) logEvent('Simulation paused for inspection.', '#ffd166')
})

showMapBtn.addEventListener('click', () => setActiveView('map'))
showGraphBtn.addEventListener('click', () => setActiveView('graph'))
showLogBtn.addEventListener('click', () => setActiveView('log'))

graphCanvasLarge.addEventListener('mousemove', (e) => {
  if (simulation.populationHistory.length < 2) return
  const rect = graphCanvasLarge.getBoundingClientRect()
  const padding = 32
  const innerW = graphCanvasLarge.width - padding * 2
  const x = ((e.clientX - rect.left) / rect.width) * graphCanvasLarge.width
  const progress = Math.max(0, Math.min(1, (x - padding) / Math.max(1, innerW)))
  graphHover = Math.round(progress * (simulation.populationHistory.length - 1))
  simulation.drawGraph()
})

graphCanvasLarge.addEventListener('mouseleave', () => {
  graphHover = null
  simulation.drawGraph()
})

startBtn.addEventListener('click', () => {
  rlMode = document.getElementById('rl-mode').checked
  targetYears = parseInt(document.getElementById('sim-years').value) || 100
  frames = 0
  currentYear = 0
  currentSeasonIdx = 0
  isPaused = false
  pauseBtn.innerText = 'Pause'
  simulation.updateSeasonUI()
  document.getElementById('year-display').innerText = `Year: 0 / ${targetYears}`

  const soc1Config = {
    strat: document.getElementById('s1-strategy').value,
    agg: parseInt(document.getElementById('s1-aggression').value),
    spd: parseFloat(document.getElementById('s1-speed').value),
    allowWomenFighters: document.getElementById('s1-women-fight').checked
  }
  const soc2Config = {
    strat: document.getElementById('s2-strategy').value,
    agg: parseInt(document.getElementById('s2-aggression').value),
    spd: parseFloat(document.getElementById('s2-speed').value),
    allowWomenFighters: document.getElementById('s2-women-fight').checked
  }

  setupPanel.classList.add('hidden')
  statsPanel.classList.remove('hidden')
  setActiveView('map')
  
  simulation.init(soc1Config, soc2Config)
  isRunning = true
  loop()
})

resetBtn.addEventListener('click', () => {
  isRunning = false
  isPaused = false
  pauseBtn.innerText = 'Pause'
  cancelAnimationFrame(animationId)
  ctx.clearRect(0, 0, width, height)
  gCtx.clearRect(0, 0, graphCanvas.width, graphCanvas.height)
  gLargeCtx.clearRect(0, 0, graphCanvasLarge.width, graphCanvasLarge.height)
  setActiveView('map')
  
  setupPanel.classList.remove('hidden')
  statsPanel.classList.add('hidden')
})

setActiveView('map')
