import test from 'node:test';
import assert from 'node:assert';
import { assertValidTransition, isValidTransition, JobStatus } from '../lib/jobs/stateMachine';
import { buildStubLayoutResponse } from '../scripts/layoutJobWorker';

const allowedTransitions: Array<[JobStatus, JobStatus]> = [
  ['queued', 'running'],
  ['queued', 'failed'],
  ['queued', 'cancelled'],
  ['running', 'completed'],
  ['running', 'failed'],
  ['running', 'cancelled'],
];

const illegalTransitions: Array<[JobStatus, JobStatus]> = [
  ['completed', 'running'],
  ['failed', 'running'],
  ['cancelled', 'running'],
  ['completed', 'queued'],
  ['failed', 'queued'],
];

test('allowed transitions are accepted', () => {
  for (const [from, to] of allowedTransitions) {
    assert.ok(isValidTransition(from, to), `${from} -> ${to} should be allowed`);
    assert.doesNotThrow(() => assertValidTransition(from, to));
  }
});

test('illegal transitions are rejected', () => {
  for (const [from, to] of illegalTransitions) {
    assert.strictEqual(isValidTransition(from, to), false, `${from} -> ${to} should be blocked`);
    assert.throws(() => assertValidTransition(from, to));
  }
});

test('worker stub state path stays valid', () => {
  const path: JobStatus[] = ['queued', 'running', 'completed'];
  for (let i = 0; i < path.length - 1; i += 1) {
    assert.doesNotThrow(() => assertValidTransition(path[i], path[i + 1]));
  }
});

test('stub layout response always contains objects', () => {
  const result = buildStubLayoutResponse(123);
  assert.ok(Array.isArray(result.objects));
  assert.ok(result.objects.length > 0, 'Should produce at least one object stub');
  assert.ok(result.world_scale > 0, 'world_scale should be defined');
});
