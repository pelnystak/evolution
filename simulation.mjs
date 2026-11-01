export const WORLD = Object.freeze({ width: 1000, height: 600, maxPopulation: 180, maxFood: 300 });

const clamp = (value, min, max) => Math.max(min, Math.min(max, value));

function randomGenerator(seed) {
  let state = seed >>> 0;
  return () => {
    state += 0x6d2b79f5;
    let value = state;
    value = Math.imul(value ^ (value >>> 15), value | 1);
    value ^= value + Math.imul(value ^ (value >>> 7), value | 61);
    return ((value ^ (value >>> 14)) >>> 0) / 4294967296;
  };
}

export class EvolutionWorld {
  constructor(settings = {}) {
    this.reset(settings);
  }

  reset(settings = {}) {
    this.settings = {
      seed: clamp(Number(settings.seed) || 0, 0, 4294967295) >>> 0,
      initialPopulation: clamp(Number(settings.initialPopulation) || 34, 10, 70),
      foodSupply: clamp(Number(settings.foodSupply) || 65, 10, 100),
      mutationChance: clamp(Number(settings.mutationChance ?? 14), 0, 40),
    };
    this.random = randomGenerator(this.settings.seed);
    this.tickCount = 0;
    this.births = 0;
    this.deaths = 0;
    this.foodBudget = 0;
    this.agents = [];
    this.food = [];
    this.history = [];

    for (let i = 0; i < this.settings.initialPopulation; i++) {
      this.agents.push(this.createAgent());
    }
    this.addFood(Math.round(55 + this.settings.foodSupply * 0.55));
    this.recordHistory();
  }

  createAgent(parent) {
    const rand = this.random;
    const mutate = (value, min, max, amount) => {
      if (rand() >= this.settings.mutationChance / 100) return value;
      return clamp(value + (rand() * 2 - 1) * amount, min, max);
    };
    return {
      x: parent ? clamp(parent.x + (rand() - 0.5) * 24, 10, WORLD.width - 10) : 20 + rand() * (WORLD.width - 40),
      y: parent ? clamp(parent.y + (rand() - 0.5) * 24, 10, WORLD.height - 10) : 20 + rand() * (WORLD.height - 40),
      angle: rand() * Math.PI * 2,
      speed: parent ? mutate(parent.speed, 0.7, 1.85, 0.18) : 0.9 + rand() * 0.3,
      sight: parent ? mutate(parent.sight, 65, 190, 20) : 92 + rand() * 30,
      energy: parent ? 38 : 55 + rand() * 15,
      age: 0,
      generation: parent ? parent.generation + 1 : 0,
      cooldown: parent ? 160 : Math.floor(rand() * 100),
    };
  }

  addFood(count) {
    const slots = Math.min(count, WORLD.maxFood - this.food.length);
    for (let i = 0; i < slots; i++) {
      this.food.push({ x: 14 + this.random() * (WORLD.width - 28), y: 14 + this.random() * (WORLD.height - 28) });
    }
  }

  tick() {
    this.tickCount++;
    this.foodBudget += 0.18 + this.settings.foodSupply * 0.0045;
    if (this.foodBudget >= 1) {
      const amount = Math.floor(this.foodBudget);
      this.addFood(amount);
      this.foodBudget -= amount;
    }

    const newborns = [];
    const survivors = [];
    for (const agent of this.agents) {
      agent.age++;
      agent.cooldown = Math.max(0, agent.cooldown - 1);

      let nearest = -1;
      let nearestDistance2 = agent.sight * agent.sight;
      for (let i = 0; i < this.food.length; i++) {
        const dx = this.food[i].x - agent.x;
        const dy = this.food[i].y - agent.y;
        const distance2 = dx * dx + dy * dy;
        if (distance2 < nearestDistance2) {
          nearestDistance2 = distance2;
          nearest = i;
        }
      }

      if (nearest >= 0) {
        const target = this.food[nearest];
        const targetAngle = Math.atan2(target.y - agent.y, target.x - agent.x);
        const difference = Math.atan2(Math.sin(targetAngle - agent.angle), Math.cos(targetAngle - agent.angle));
        agent.angle += clamp(difference, -0.16, 0.16);
      } else {
        agent.angle += (this.random() - 0.5) * 0.23;
      }

      agent.x = clamp(agent.x + Math.cos(agent.angle) * agent.speed * 1.75, 10, WORLD.width - 10);
      agent.y = clamp(agent.y + Math.sin(agent.angle) * agent.speed * 1.75, 10, WORLD.height - 10);
      if (agent.x === 10 || agent.x === WORLD.width - 10 || agent.y === 10 || agent.y === WORLD.height - 10) {
        agent.angle += Math.PI * (0.65 + this.random() * 0.7);
      }

      if (nearest >= 0 && nearestDistance2 < 110) {
        agent.energy = Math.min(110, agent.energy + 22);
        this.food.splice(nearest, 1);
      }
      agent.energy -= 0.075 + agent.speed * 0.043 + agent.sight * 0.00025;

      if (agent.energy <= 0 || agent.age >= 1650) {
        this.deaths++;
        continue;
      }
      if (agent.age > 130 && agent.cooldown === 0 && agent.energy >= 90 &&
          this.agents.length + newborns.length < WORLD.maxPopulation) {
        agent.energy -= 43;
        agent.cooldown = 165;
        newborns.push(this.createAgent(agent));
        this.births++;
      }
      survivors.push(agent);
    }
    this.agents = survivors.concat(newborns);
    if (this.tickCount % 15 === 0) this.recordHistory();
  }

  snapshot() {
    const population = this.agents.length;
    const sum = (key) => this.agents.reduce((total, agent) => total + agent[key], 0);
    return {
      tick: this.tickCount,
      population,
      food: this.food.length,
      generation: population ? Math.max(...this.agents.map((agent) => agent.generation)) : 0,
      speed: population ? sum('speed') / population : 0,
      sight: population ? sum('sight') / population : 0,
      births: this.births,
      deaths: this.deaths,
    };
  }

  recordHistory() {
    this.history.push(this.snapshot());
    if (this.history.length > 180) this.history.shift();
  }
}
