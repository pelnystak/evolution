import test from 'node:test';
import assert from 'node:assert/strict';
import { EvolutionWorld, WORLD } from './simulation.mjs';

test('the same seed and settings reproduce the same history', () => {
  const first = new EvolutionWorld({ seed: 321, foodSupply: 40, mutationChance: 25 });
  const second = new EvolutionWorld({ seed: 321, foodSupply: 40, mutationChance: 25 });
  for (let i = 0; i < 900; i++) {
    first.tick();
    second.tick();
  }
  assert.deepEqual(first.history, second.history);
  assert.deepEqual(first.snapshot(), second.snapshot());
});

test('population, resources and traits stay bounded during a long rich-food run', () => {
  const world = new EvolutionWorld({ seed: 17, foodSupply: 100, mutationChance: 40, initialPopulation: 70 });
  for (let i = 0; i < 3600; i++) {
    world.tick();
    assert.ok(world.agents.length <= WORLD.maxPopulation);
    assert.ok(world.food.length <= WORLD.maxFood);
  }
  for (const agent of world.agents) {
    assert.ok(agent.x >= 10 && agent.x <= WORLD.width - 10);
    assert.ok(agent.y >= 10 && agent.y <= WORLD.height - 10);
    assert.ok(agent.speed >= 0.7 && agent.speed <= 1.85);
    assert.ok(agent.sight >= 65 && agent.sight <= 190);
    assert.ok(Number.isFinite(agent.energy));
  }
});

test('reset restores the initial state and clears the old run', () => {
  const settings = { seed: 42, foodSupply: 65, mutationChance: 14, initialPopulation: 34 };
  const world = new EvolutionWorld(settings);
  for (let i = 0; i < 400; i++) world.tick();
  world.reset(settings);
  const fresh = new EvolutionWorld(settings);
  assert.deepEqual(world.agents, fresh.agents);
  assert.deepEqual(world.food, fresh.food);
  assert.deepEqual(world.snapshot(), fresh.snapshot());
  assert.equal(world.history.length, 1);
});
