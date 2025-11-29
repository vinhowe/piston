import { openDb, txRequest } from '$lib/dataUtils';
import { SvelteMap } from 'svelte/reactivity';

import type { Config } from './config';
import type { RunData, StepData } from './runs.svelte';

export type SavedRun = Omit<RunData, 'metrics'> & { metrics: Record<string, StepData[]> };

const DB_NAME = 'piston-last-session-store';
const DB_VERSION = 1;
const STORE_SESSION = 'session';
const STORE_CHECKPOINT = 'checkpoint';
const STORE_META = 'meta';
const META_LAST_RUN_ID_KEY = 'lastRunId';

export function serializeRun(run: RunData): SavedRun {
	const metrics = Object.fromEntries([...run.metrics.entries()].map(([k, v]) => [k, v.data]));
	return { ...run, metrics };
}

export function deserializeRun(saved: SavedRun): RunData {
	return {
		...saved,
		metrics: new SvelteMap(
			Object.entries(saved.metrics).map(([k, v]) => [k, { metricName: k, data: v }])
		)
	};
}

class LastSessionStore {
	private dbPromise: Promise<IDBDatabase> | null = null;

	private get db(): Promise<IDBDatabase> {
		if (!this.dbPromise)
			this.dbPromise = openDb(DB_NAME, DB_VERSION, (db) => {
				if (!db.objectStoreNames.contains(STORE_SESSION)) db.createObjectStore(STORE_SESSION);
				if (!db.objectStoreNames.contains(STORE_CHECKPOINT)) db.createObjectStore(STORE_CHECKPOINT);
				if (!db.objectStoreNames.contains(STORE_META)) db.createObjectStore(STORE_META);
			});
		return this.dbPromise;
	}

	private async getLastRunId(db: IDBDatabase): Promise<string | undefined> {
		return txRequest<string | undefined>(db, STORE_META, 'readonly', (s) =>
			s.get(META_LAST_RUN_ID_KEY)
		);
	}

	async get(): Promise<{ run: RunData; checkpoint: Uint8Array } | null> {
		const db = await this.db;
		const runId = await this.getLastRunId(db);
		if (!runId) return null;
		const [run, buffer] = await Promise.all([
			txRequest<SavedRun | undefined>(db, STORE_SESSION, 'readonly', (s) => s.get(runId)),
			txRequest<ArrayBuffer | undefined>(db, STORE_CHECKPOINT, 'readonly', (s) => s.get(runId))
		]);
		if (!run || !buffer) return null;
		return { run: deserializeRun(run), checkpoint: new Uint8Array(buffer) };
	}

	async set(run: RunData, checkpoint: Uint8Array | ArrayBuffer): Promise<void> {
		const db = await this.db;
		const buf = checkpoint instanceof Uint8Array ? (checkpoint.buffer as ArrayBuffer) : checkpoint;
		// Clear all existing sessions; we'll want to remove this if we ever support multiple
		// persistence.
		await this.delete();
		await Promise.all([
			txRequest(db, STORE_SESSION, 'readwrite', (s) => s.put(serializeRun(run), run.runId)),
			txRequest(db, STORE_CHECKPOINT, 'readwrite', (s) => s.put(buf, run.runId)),
			txRequest(db, STORE_META, 'readwrite', (s) => s.put(run.runId, META_LAST_RUN_ID_KEY))
		]);
	}

	/**
	 * Update only the config of the last run in storage without touching other data.
	 * This avoids fully deserializing metrics and only rewrites the config field.
	 */
	async updateConfig(mutator: (config: Config) => Config | void): Promise<void> {
		const db = await this.db;
		const runId = await this.getLastRunId(db);
		if (!runId) return;

		const saved = await txRequest<SavedRun | undefined>(db, STORE_SESSION, 'readonly', (s) =>
			s.get(runId)
		);
		if (!saved) return;

		const maybeNew = mutator(saved.config as Config);
		const newConfig = (maybeNew ? maybeNew : saved.config) as Config;
		const updated: SavedRun = { ...saved, config: newConfig };
		await txRequest(db, STORE_SESSION, 'readwrite', (s) => s.put(updated, runId));
	}

	async exists(): Promise<boolean> {
		const db = await this.db;
		const runId = await this.getLastRunId(db);
		if (!runId) return false;
		const [run, buffer] = await Promise.all([
			txRequest<SavedRun | undefined>(db, STORE_SESSION, 'readonly', (s) => s.get(runId)),
			txRequest<ArrayBuffer | undefined>(db, STORE_CHECKPOINT, 'readonly', (s) => s.get(runId))
		]);
		return run != null && buffer != null;
	}

	async delete(): Promise<void> {
		const db = await this.db;
		await Promise.all([
			txRequest(db, STORE_SESSION, 'readwrite', (s) => s.clear()),
			txRequest(db, STORE_CHECKPOINT, 'readwrite', (s) => s.clear()),
			txRequest(db, STORE_META, 'readwrite', (s) => s.delete(META_LAST_RUN_ID_KEY))
		]);
	}
}

export const lastSessionStore = new LastSessionStore();
