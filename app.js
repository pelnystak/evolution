import { EvolutionWorld } from './simulation.mjs';

const $ = (id) => document.getElementById(id);
const worldCanvas = $('worldCanvas');
const worldContext = worldCanvas.getContext('2d');
const populationContext = $('populationChart').getContext('2d');
const traitsContext = $('traitsChart').getContext('2d');
const controls = {
  food: $('foodInput'), mutation: $('mutationInput'), population: $('populationInput'), seed: $('seedInput'),
  play: $('playButton'), step: $('stepButton'), reset: $('resetButton'),
};
const colors = { green: '#c6f278', orange: '#ff9a61', lavender: '#c6b7ff', ink: '#f6f2e8', quiet: '#74837a' };
let world;
let running = false;
let hasStarted = false;
let pace = 1;
let lastFrame = 0;
let accumulator = 0;

function applyMode(mode) {
  const day = mode === 'day';
  document.documentElement.dataset.mode = day ? 'day' : 'night';
  $('modeToggle').textContent = day ? '☾' : '☼';
  $('modeToggle').setAttribute('aria-pressed', String(day));
  $('modeToggle').setAttribute('aria-label', day ? 'Switch to dark mode' : 'Switch to light mode');
  document.querySelector('meta[name="theme-color"]').content = day ? '#e9e6d9' : '#172321';
  colors.green = day ? '#617d48' : '#c6f278';
  colors.orange = day ? '#bf6845' : '#ff9a61';
  colors.lavender = day ? '#7663a6' : '#c6b7ff';
  colors.quiet = day ? '#738071' : '#74837a';
  if (world) render();
}

let savedMode = 'night';
try { savedMode = localStorage.getItem('pj-lab-mode') || 'night'; } catch { /* Private browsing can block storage. */ }
applyMode(savedMode);
$('modeToggle').addEventListener('click', () => {
  const next = document.documentElement.dataset.mode === 'day' ? 'night' : 'day';
  try { localStorage.setItem('pj-lab-mode', next); } catch { /* Keep the in-page choice. */ }
  applyMode(next);
});

function settingsFromControls() {
  return {
    initialPopulation: Number(controls.population.value),
    foodSupply: Number(controls.food.value),
    mutationChance: Number(controls.mutation.value),
    seed: Number(controls.seed.value),
  };
}

function updateControlLabels() {
  $('foodValue').value = controls.food.value;
  $('mutationValue').value = `${controls.mutation.value}%`;
  $('populationValue').value = controls.population.value;
}

function setRunning(value) {
  running = value && world.agents.length > 0;
  if (running) hasStarted = true;
  controls.play.innerHTML = running ? 'Ⅱ <span>Pause simulation</span>' : `▶ <span>${hasStarted ? 'Resume simulation' : 'Start simulation'}</span>`;
  controls.step.disabled = running;
  $('statusBadge').textContent = world.agents.length === 0 ? 'EXTINCT' : running ? 'RUNNING' : 'PAUSED';
  $('statusBadge').classList.toggle('is-running', running);
  lastFrame = 0;
}

function reset() {
  world = new EvolutionWorld(settingsFromControls());
  hasStarted = false;
  setRunning(false);
  accumulator = 0;
  render();
}

function drawWorld() {
  const ctx = worldContext;
  const { width, height } = worldCanvas;
  ctx.fillStyle = '#172321';
  ctx.fillRect(0, 0, width, height);
  ctx.strokeStyle = 'rgba(224,242,223,.055)';
  ctx.lineWidth = 1;
  for (let x = 0; x <= width; x += 50) { ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, height); ctx.stroke(); }
  for (let y = 0; y <= height; y += 50) { ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(width, y); ctx.stroke(); }
  ctx.fillStyle = 'rgba(198,242,120,.16)';
  for (const food of world.food) {
    ctx.beginPath(); ctx.arc(food.x, food.y, 5, 0, Math.PI * 2); ctx.fill();
  }
  ctx.fillStyle = colors.green;
  for (const food of world.food) {
    ctx.beginPath(); ctx.arc(food.x, food.y, 2.1, 0, Math.PI * 2); ctx.fill();
  }
  for (const agent of world.agents) {
    const hue = 101 - (agent.speed - 0.7) / 1.15 * 65;
    const radius = 5.5 + agent.energy / 40;
    ctx.fillStyle = `hsla(${hue}, 85%, 74%, .13)`;
    ctx.beginPath(); ctx.arc(agent.x, agent.y, radius + 7, 0, Math.PI * 2); ctx.fill();
    ctx.fillStyle = `hsl(${hue}, 84%, 71%)`;
    ctx.beginPath(); ctx.arc(agent.x, agent.y, radius, 0, Math.PI * 2); ctx.fill();
    ctx.fillStyle = '#172321';
    ctx.beginPath(); ctx.arc(agent.x + Math.cos(agent.angle) * radius * .42, agent.y + Math.sin(agent.angle) * radius * .42, 1.5, 0, Math.PI * 2); ctx.fill();
  }
  if (world.agents.length === 0) {
    ctx.fillStyle = 'rgba(23,35,33,.72)'; ctx.fillRect(0, 0, width, height);
    ctx.fillStyle = colors.ink; ctx.textAlign = 'center'; ctx.font = '500 30px Avenir Next, sans-serif';
    ctx.fillText('The population went extinct.', width / 2, height / 2 - 10);
    ctx.fillStyle = colors.quiet; ctx.font = '15px Avenir Next, sans-serif';
    ctx.fillText('Increase food supply or reset the world.', width / 2, height / 2 + 24);
    ctx.textAlign = 'start';
  }
}

function drawChart(ctx, history, series, maximum) {
  const { width, height } = ctx.canvas;
  ctx.clearRect(0, 0, width, height);
  const left = 28, right = width - 12, top = 12, bottom = height - 24;
  ctx.strokeStyle = 'rgba(238,244,226,.1)'; ctx.lineWidth = 1;
  for (let i = 0; i <= 4; i++) {
    const y = top + (bottom - top) * i / 4;
    ctx.beginPath(); ctx.moveTo(left, y); ctx.lineTo(right, y); ctx.stroke();
  }
  for (const item of series) {
    ctx.strokeStyle = item.color; ctx.lineWidth = 2.4; ctx.lineJoin = 'round'; ctx.beginPath();
    history.forEach((point, index) => {
      const x = left + index / Math.max(1, history.length - 1) * (right - left);
      const y = bottom - Math.min(1, Math.max(0, item.value(point) / maximum)) * (bottom - top);
      if (index === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    });
    ctx.stroke();
    if (history.length === 1) {
      ctx.fillStyle = item.color; ctx.beginPath(); ctx.arc(left, bottom - item.value(history[0]) / maximum * (bottom - top), 3.5, 0, Math.PI * 2); ctx.fill();
    }
  }
  ctx.font = '10px SFMono-Regular, Consolas, monospace'; ctx.fillStyle = colors.quiet;
  ctx.fillText('START', left, height - 4); ctx.textAlign = 'right'; ctx.fillText('NOW', right, height - 4); ctx.textAlign = 'start';
}

function render() {
  const stat = world.snapshot();
  $('populationStat').textContent = stat.population;
  $('generationStat').textContent = stat.generation;
  $('speedStat').textContent = stat.speed.toFixed(2);
  $('sightStat').textContent = Math.round(stat.sight);
  $('birthsStat').textContent = stat.births;
  $('foodStat').textContent = stat.food;
  $('tickLabel').textContent = `DAY ${Math.floor(stat.tick / 30)} · TICK ${stat.tick}`;
  if (stat.population === 0 && running) setRunning(false);
  drawWorld();
  const history = world.history;
  drawChart(populationContext, history, [
    { value: (point) => point.population, color: colors.green },
    { value: (point) => point.food, color: colors.orange },
  ], Math.max(120, ...history.map((point) => Math.max(point.population, point.food))) * 1.12);
  drawChart(traitsContext, history, [
    { value: (point) => (point.speed - 0.7) / 1.15, color: colors.lavender },
    { value: (point) => (point.sight - 65) / 125, color: colors.orange },
  ], 1);
}

function frame(time) {
  if (running) {
    if (lastFrame) accumulator += Math.min(time - lastFrame, 100) * pace;
    lastFrame = time;
    let steps = 0;
    while (accumulator >= 33.33 && steps < 12 && world.agents.length > 0) {
      world.tick(); accumulator -= 33.33; steps++;
    }
    if (steps) render();
    if (steps === 12) accumulator = 0;
  }
  requestAnimationFrame(frame);
}

controls.play.addEventListener('click', () => setRunning(!running));
controls.step.addEventListener('click', () => { if (!running && world.agents.length > 0) { world.tick(); render(); } });
controls.reset.addEventListener('click', reset);
for (const button of document.querySelectorAll('[data-pace]')) {
  button.addEventListener('click', () => {
    pace = Number(button.dataset.pace);
    for (const item of document.querySelectorAll('[data-pace]')) {
      const selected = item === button;
      item.classList.toggle('selected', selected);
      item.setAttribute('aria-pressed', String(selected));
    }
  });
}
controls.food.addEventListener('input', () => { updateControlLabels(); world.settings.foodSupply = Number(controls.food.value); });
controls.mutation.addEventListener('input', () => { updateControlLabels(); world.settings.mutationChance = Number(controls.mutation.value); });
controls.population.addEventListener('input', updateControlLabels);
updateControlLabels();
reset();
requestAnimationFrame(frame);
