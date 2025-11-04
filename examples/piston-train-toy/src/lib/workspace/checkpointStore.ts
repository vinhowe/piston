import { openDb, txRequest } from '$lib/dataUtils';

const DB_NAME = 'piston-checkpoint-store';
const DB_VERSION = 1;
const STORE_CHECKPOINTS = 'checkpoints';

export class CheckpointStore {
	private dbPromise: Promise<IDBDatabase> | null = null;

	private get db(): Promise<IDBDatabase> {
		if (!this.dbPromise)
			this.dbPromise = openDb(DB_NAME, DB_VERSION, (db) => {
				if (!db.objectStoreNames.contains(STORE_CHECKPOINTS)) {
					db.createObjectStore(STORE_CHECKPOINTS);
				}
			});
		return this.dbPromise;
	}

	async get(runId: string): Promise<ArrayBuffer | undefined> {
		const db = await this.db;
		return txRequest<ArrayBuffer | undefined>(db, STORE_CHECKPOINTS, 'readonly', (s) =>
			s.get(runId)
		);
	}

	async set(runId: string, bytes: Uint8Array | ArrayBuffer): Promise<void> {
		const db = await this.db;
		const buf = bytes instanceof Uint8Array ? (bytes.buffer as ArrayBuffer) : bytes;
		await txRequest(db, STORE_CHECKPOINTS, 'readwrite', (s) => s.put(buf, runId));
	}

	async has(runId: string): Promise<boolean> {
		const db = await this.db;
		const res = await txRequest<ArrayBuffer | undefined>(db, STORE_CHECKPOINTS, 'readonly', (s) =>
			s.get(runId)
		);
		return res != null;
	}

	async delete(runId: string): Promise<void> {
		const db = await this.db;
		await txRequest(db, STORE_CHECKPOINTS, 'readwrite', (s) => s.delete(runId));
	}

	async clear(): Promise<void> {
		const db = await this.db;
		await txRequest(db, STORE_CHECKPOINTS, 'readwrite', (s) => s.clear());
	}
}

export const checkpointStore = new CheckpointStore();
