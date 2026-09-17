/**
 * Deleted-thread write guards (https://github.com/mastra-ai/mastra/issues/23177)
 *
 * An observational-memory cycle keeps running after the LLM call that produced its
 * observations. If the thread is deleted during that call, the cycle used to persist
 * into rows and vector indexes that `Memory.deleteThread` had already cleaned up —
 * leaving the deleted thread's text permanently retrievable, resurrecting its OM
 * record, or throwing `Observational memory record not found` from a background path.
 *
 * The interleave is made deterministic by having the mock observer delete the thread
 * from inside `doStream`, which the observer runner awaits before `process()`/`persist()`
 * run. No timers, no polling, no reliance on scheduling luck.
 */

import { MockLanguageModelV2, convertArrayToReadableStream } from '@internal/ai-sdk-v5/test';
import type { MastraDBMessage, MastraMessageContentV2 } from '@mastra/core/agent';
import { InMemoryMemory, InMemoryDB } from '@mastra/core/storage';
import { describe, it, expect, beforeEach, vi } from 'vitest';

import { BufferingCoordinator } from '../buffering-coordinator';
import { Extractor } from '../extractor';
import { ResourceScopedObservationStrategy } from '../observation-strategies/resource-scoped';
import { ObservationalMemory } from '../observational-memory';

const OBSERVATION_TEXT = `<observations>
* The deploy key lives at /etc/secrets/deploy-key
</observations>
<current-task>
- Primary: Continue conversation
</current-task>`;

/**
 * Resource-scoped observer output. The multi-thread prompt requires each thread's
 * observations nested in a `<thread id="...">` block, and `process()` matches those
 * ids back to the real threads, so the id has to be the live thread id.
 */
function resourceObservationText(threadId: string): string {
  return `<observations>
<thread id="${threadId}">
* The deploy key lives at /etc/secrets/deploy-key
</thread>
</observations>
<current-task>
- Primary: Continue conversation
</current-task>`;
}

function createTestMessage(
  content: string,
  role: 'user' | 'assistant' = 'user',
  id?: string,
  createdAt?: Date,
): MastraDBMessage {
  return {
    id: id ?? `msg-${Math.random().toString(36).slice(2)}`,
    role,
    content: { format: 2, parts: [{ type: 'text', text: content }] } as MastraMessageContentV2,
    type: 'text',
    createdAt: createdAt ?? new Date(),
  };
}

/** Generate N messages padded to comfortably exceed the configured token thresholds. */
function createBulkMessages(count: number, threadId: string): MastraDBMessage[] {
  const base = Date.now() - count * 1000;
  return Array.from({ length: count }, (_, i) => ({
    ...createTestMessage(
      `Message ${i}: `.padEnd(200, 'x'),
      i % 2 === 0 ? 'user' : 'assistant',
      `${threadId}-msg-${i}`,
      new Date(base + i * 1000),
    ),
    threadId,
  }));
}

/**
 * Observer model that runs `onCall` while the LLM request is in flight — the point at
 * which a real `deleteThread` would interleave.
 */
function createObserverModel(onCall?: () => Promise<void>, text: string = OBSERVATION_TEXT) {
  const runHook = async () => {
    if (onCall) await onCall();
  };

  return new MockLanguageModelV2({
    doGenerate: async () => {
      await runHook();
      return {
        rawCall: { rawPrompt: null, rawSettings: {} },
        finishReason: 'stop',
        usage: { inputTokens: 100, outputTokens: 50, totalTokens: 150 },
        warnings: [],
        content: [{ type: 'text', text }],
      };
    },
    doStream: async () => {
      await runHook();
      return {
        stream: convertArrayToReadableStream([
          { type: 'stream-start', warnings: [] },
          { type: 'response-metadata', id: 'obs-1', modelId: 'mock-observer', timestamp: new Date() },
          { type: 'text-start', id: 'text-1' },
          { type: 'text-delta', id: 'text-1', delta: text },
          { type: 'text-end', id: 'text-1' },
          {
            type: 'finish',
            finishReason: 'stop',
            usage: { inputTokens: 100, outputTokens: 50, totalTokens: 150 },
          },
        ]),
        rawCall: { rawPrompt: null, rawSettings: {} },
        warnings: [],
      };
    },
  } as never);
}

function createOM(
  storage: InMemoryMemory,
  opts: {
    model: unknown;
    onIndexObservations: ReturnType<typeof vi.fn>;
    /** Low observation threshold so the sync path observes; buffered path uses bufferTokens. */
    messageTokens?: number;
    bufferTokens?: number | false;
    scope?: 'thread' | 'resource';
    /** Extra extractors, e.g. a `mode: 'hook'` extractor used to interleave a deletion. */
    extractors?: Extractor<any>[];
  },
) {
  return new ObservationalMemory({
    storage,
    scope: opts.scope ?? 'thread',
    retrieval: { vector: true },
    onIndexObservations: opts.onIndexObservations,
    observation: {
      model: opts.model as never,
      messageTokens: opts.messageTokens ?? 50,
      bufferTokens: opts.bufferTokens ?? false,
      extract: opts.extractors,
    },
    reflection: { model: createObserverModel() as never, observationTokens: 10_000_000 },
  });
}

/** Mirrors the destructive half of `Memory.deleteStoredThread`. */
async function simulateThreadDeletion(storage: InMemoryMemory, threadId: string, resourceId: string) {
  await storage.deleteThread({ threadId });
  await storage.clearObservationalMemory(threadId, resourceId);
}

async function seedThread(storage: InMemoryMemory, threadId: string, resourceId: string, messageCount = 5) {
  await storage.saveThread({
    thread: {
      id: threadId,
      resourceId,
      title: 'Write guard probe',
      metadata: {},
      createdAt: new Date(),
      updatedAt: new Date(),
    },
  });
  await storage.saveMessages({
    messages: createBulkMessages(messageCount, threadId).map(message => ({ ...message, resourceId })),
  });
}

// Static maps leak across tests in this package (`isolate: false` in vitest.config.ts).
beforeEach(() => {
  BufferingCoordinator.asyncBufferingOps.clear();
  BufferingCoordinator.lastBufferedBoundary.clear();
  BufferingCoordinator.lastBufferedAtTime.clear();
  BufferingCoordinator.reflectionBufferCycleIds.clear();
});

describe('deleted-thread write guards', () => {
  let storage: InMemoryMemory;
  const threadId = 'guard-thread';
  const resourceId = 'guard-resource';

  beforeEach(() => {
    storage = new InMemoryMemory({ db: new InMemoryDB() });
  });

  describe('sync observation cycle', () => {
    it('writes no observation vectors and does not resurrect the record when the thread is deleted mid-cycle', async () => {
      const onIndexObservations = vi.fn().mockResolvedValue(undefined);
      const updateActive = vi.spyOn(storage, 'updateActiveObservations');
      const initialize = vi.spyOn(storage, 'initializeObservationalMemory');
      await seedThread(storage, threadId, resourceId);

      const om = createOM(storage, {
        model: createObserverModel(() => simulateThreadDeletion(storage, threadId, resourceId)),
        onIndexObservations,
      });
      const messages = createBulkMessages(5, threadId).map(message => ({ ...message, resourceId }));

      // Must not reject: persisting into the cleared record used to throw
      // `Observational memory record not found` out of this path.
      const result = await om.observe({ threadId, resourceId, messages });

      expect(onIndexObservations).not.toHaveBeenCalled();
      expect(updateActive).not.toHaveBeenCalled();
      // The record is not recreated for a thread that no longer exists.
      expect(initialize).toHaveBeenCalledTimes(1);
      expect(await storage.getObservationalMemory(threadId, resourceId)).toBeNull();
      expect(await storage.getThreadById({ threadId })).toBeNull();
      // observe() still returns a usable record — callers read `.record.id`.
      expect(result.record).toBeTruthy();
      expect(result.reflected).toBe(false);
    });

    it('still indexes observations when the thread survives the cycle', async () => {
      const onIndexObservations = vi.fn().mockResolvedValue(undefined);
      const updateActive = vi.spyOn(storage, 'updateActiveObservations');
      await seedThread(storage, threadId, resourceId);

      const om = createOM(storage, { model: createObserverModel(), onIndexObservations });
      const messages = createBulkMessages(5, threadId).map(message => ({ ...message, resourceId }));

      await om.observe({ threadId, resourceId, messages });

      expect(onIndexObservations).toHaveBeenCalled();
      expect(updateActive).toHaveBeenCalled();
      expect(await storage.getObservationalMemory(threadId, resourceId)).toBeTruthy();
      const indexed = onIndexObservations.mock.calls.map(call => call[0]);
      expect(indexed.every(entry => entry.threadId === threadId && entry.resourceId === resourceId)).toBe(true);
      expect(indexed.some(entry => String(entry.text).includes('deploy-key'))).toBe(true);
    });
  });

  describe('buffered observation cycle', () => {
    it('writes no buffered chunk and no vectors when the thread is deleted mid-cycle', async () => {
      const onIndexObservations = vi.fn().mockResolvedValue(undefined);
      const updateBuffered = vi.spyOn(storage, 'updateBufferedObservations');
      await seedThread(storage, threadId, resourceId);

      const om = createOM(storage, {
        model: createObserverModel(() => simulateThreadDeletion(storage, threadId, resourceId)),
        onIndexObservations,
        messageTokens: 500,
        bufferTokens: 0.2,
      });

      await om.buffer({ threadId, resourceId });
      await om.waitForBuffering(threadId, resourceId, 5000);

      expect(onIndexObservations).not.toHaveBeenCalled();
      expect(updateBuffered).not.toHaveBeenCalled();
      expect(await storage.getThreadById({ threadId })).toBeNull();
    });

    it('still buffers and indexes when the thread survives the cycle', async () => {
      const onIndexObservations = vi.fn().mockResolvedValue(undefined);
      const updateBuffered = vi.spyOn(storage, 'updateBufferedObservations');
      await seedThread(storage, threadId, resourceId);

      const om = createOM(storage, {
        model: createObserverModel(),
        onIndexObservations,
        messageTokens: 500,
        bufferTokens: 0.2,
      });

      await om.buffer({ threadId, resourceId });
      await om.waitForBuffering(threadId, resourceId, 5000);

      expect(onIndexObservations).toHaveBeenCalled();
      expect(updateBuffered).toHaveBeenCalled();
      const status = await om.getStatus({ threadId, resourceId });
      expect(status.bufferedChunkCount).toBe(1);
    });
  });

  describe('resource-scoped observation cycle', () => {
    // In resource scope the observational-memory record is keyed by resource, and
    // `clearObservationalMemory(threadId, resourceId)` only clears the thread-keyed
    // record — so the record-keyed guard the sync/async-buffer paths use still reads
    // as live here. These two cases pin the per-thread liveness check that covers it.
    const resourceScopedObs = () => resourceObservationText(threadId);

    it('does not add a deleted thread to the shared resource record when the thread is deleted mid-cycle', async () => {
      const onIndexObservations = vi.fn().mockResolvedValue(undefined);
      const updateActive = vi.spyOn(storage, 'updateActiveObservations');
      const initialize = vi.spyOn(storage, 'initializeObservationalMemory');
      await seedThread(storage, threadId, resourceId);

      const om = createOM(storage, {
        model: createObserverModel(() => simulateThreadDeletion(storage, threadId, resourceId), resourceScopedObs()),
        onIndexObservations,
        scope: 'resource',
      });
      const messages = createBulkMessages(5, threadId).map(message => ({ ...message, resourceId }));

      const result = await om.observe({ threadId, resourceId, messages });

      // The thread-keyed record is gone and stays gone.
      expect(await storage.getThreadById({ threadId })).toBeNull();
      expect(await storage.getObservationalMemory(threadId, resourceId)).toBeNull();

      // The resource-keyed record survives — it is shared by every remaining thread —
      // but must not carry the deleted thread's text or its section.
      const resourceRecord = await storage.getObservationalMemory(null, resourceId);
      expect(resourceRecord).toBeTruthy();
      expect(resourceRecord!.activeObservations).not.toContain('deploy-key');
      expect(resourceRecord!.activeObservations).not.toContain(`<thread id="${threadId}">`);
      expect(onIndexObservations).not.toHaveBeenCalled();
      // observe() still returns a usable record — callers read `.record.id`.
      expect(result.record).toBeTruthy();
      expect(result.reflected).toBe(false);
      // The record was created once, inside the lock, and not re-created by the cycle.
      expect(initialize).toHaveBeenCalledTimes(1);
      // With no surviving threads there is nothing to append, so the shared record is
      // left untouched rather than stamped with a fresh `lastObservedAt`.
      expect(updateActive).not.toHaveBeenCalled();
    });

    it('still adds the thread to the shared resource record when the thread survives the cycle', async () => {
      const onIndexObservations = vi.fn().mockResolvedValue(undefined);
      await seedThread(storage, threadId, resourceId);

      const om = createOM(storage, {
        model: createObserverModel(undefined, resourceScopedObs()),
        onIndexObservations,
        scope: 'resource',
      });
      const messages = createBulkMessages(5, threadId).map(message => ({ ...message, resourceId }));

      await om.observe({ threadId, resourceId, messages });

      const resourceRecord = await storage.getObservationalMemory(null, resourceId);
      expect(resourceRecord).toBeTruthy();
      expect(resourceRecord!.activeObservations).toContain('deploy-key');
      expect(resourceRecord!.activeObservations).toContain(`<thread id="${threadId}">`);
      expect(await storage.getThreadById({ threadId })).toBeTruthy();
    });

    it('does not add a thread that is deleted while an extractor hook is running', async () => {
      // The extractor hook runs after the observer call and is awaited, so a deletion
      // inside it lands between the liveness check at the top of `process()` and the
      // point the thread's observations would be kept.
      const onIndexObservations = vi.fn().mockResolvedValue(undefined);
      const updateActive = vi.spyOn(storage, 'updateActiveObservations');
      await seedThread(storage, threadId, resourceId);

      const om = createOM(storage, {
        model: createObserverModel(undefined, resourceScopedObs()),
        onIndexObservations,
        scope: 'resource',
        extractors: [
          new Extractor({
            name: 'probe-delete-thread',
            mode: 'hook',
            onExtracted: async () => {
              await simulateThreadDeletion(storage, threadId, resourceId);
            },
          }),
        ],
      });
      const messages = createBulkMessages(5, threadId).map(message => ({ ...message, resourceId }));

      await om.observe({ threadId, resourceId, messages });

      expect(await storage.getThreadById({ threadId })).toBeNull();
      const resourceRecord = await storage.getObservationalMemory(null, resourceId);
      expect(resourceRecord).toBeTruthy();
      expect(resourceRecord!.activeObservations).not.toContain('deploy-key');
      expect(resourceRecord!.activeObservations).not.toContain(`<thread id="${threadId}">`);
      expect(onIndexObservations).not.toHaveBeenCalled();
      expect(updateActive).not.toHaveBeenCalled();
    });

    it('does not add a thread deleted between process() and persist()', async () => {
      // `process()` merges the record text; `persist()` writes it. A deletion in that gap
      // is only covered by persist()'s own liveness check, which rebuilds the text.
      const onIndexObservations = vi.fn().mockResolvedValue(undefined);
      const updateActive = vi.spyOn(storage, 'updateActiveObservations');
      await seedThread(storage, threadId, resourceId);

      const originalProcess = ResourceScopedObservationStrategy.prototype.process;
      const processSpy = vi
        .spyOn(ResourceScopedObservationStrategy.prototype, 'process')
        .mockImplementation(async function (this: ResourceScopedObservationStrategy, ...args) {
          const result = await originalProcess.apply(this, args);
          await simulateThreadDeletion(storage, threadId, resourceId);
          return result;
        });

      try {
        const om = createOM(storage, {
          model: createObserverModel(undefined, resourceScopedObs()),
          onIndexObservations,
          scope: 'resource',
        });
        const messages = createBulkMessages(5, threadId).map(message => ({ ...message, resourceId }));

        await om.observe({ threadId, resourceId, messages });
      } finally {
        processSpy.mockRestore();
      }

      expect(await storage.getThreadById({ threadId })).toBeNull();
      const resourceRecord = await storage.getObservationalMemory(null, resourceId);
      expect(resourceRecord).toBeTruthy();
      expect(resourceRecord!.activeObservations).not.toContain('deploy-key');
      expect(resourceRecord!.activeObservations).not.toContain(`<thread id="${threadId}">`);
      expect(onIndexObservations).not.toHaveBeenCalled();
      expect(updateActive).not.toHaveBeenCalled();
    });
  });
});
