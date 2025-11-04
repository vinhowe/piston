// A tiny IndexedDB-backed LRU cache specialized for dataset shard ArrayBuffers.
// Uses one object store for blobs and a metadata store for LRU bookkeeping.

import { openDb, txRequest } from '$lib/dataUtils';

const DB_NAME = 'natural-shard-cache';
const DB_VERSION = 1;
const STORE_BLOBS = 'blobs';
const META_KEY = 'lru';
const STORE_META = 'meta';

export interface ShardCacheOptions {
	// Maximum total size in bytes
	maxBytes?: number;
}

type LruEntry = { last: number; size: number };
type LruState = { entries: Record<string, LruEntry>; totalSize: number };

const DEFAULTS: Required<ShardCacheOptions> = {
	maxBytes: 256 * 1024 * 1024 // 256MB
};

async function readMeta(db: IDBDatabase): Promise<LruState> {
	const res = await txRequest<LruState | undefined>(db, STORE_META, 'readonly', (s) =>
		s.get(META_KEY)
	);
	return res ?? { entries: {}, totalSize: 0 };
}

async function writeMeta(db: IDBDatabase, meta: LruState): Promise<void> {
	await txRequest(db, STORE_META, 'readwrite', (s) => s.put(meta, META_KEY));
}

async function getBlob(db: IDBDatabase, url: string): Promise<ArrayBuffer | undefined> {
	return txRequest<ArrayBuffer | undefined>(db, STORE_BLOBS, 'readonly', (s) => s.get(url));
}

async function putBlob(db: IDBDatabase, url: string, buf: ArrayBuffer): Promise<void> {
	await txRequest(db, STORE_BLOBS, 'readwrite', (s) => s.put(buf, url));
}

async function deleteBlob(db: IDBDatabase, url: string): Promise<void> {
	await txRequest(db, STORE_BLOBS, 'readwrite', (s) => s.delete(url));
}

export class ShardCache {
	private dbPromise: Promise<IDBDatabase> | null = null;
	private options: Required<ShardCacheOptions>;

	constructor(options: ShardCacheOptions = {}) {
		this.options = { ...DEFAULTS, ...options };
	}

	private get db(): Promise<IDBDatabase> {
		if (!this.dbPromise)
			this.dbPromise = openDb(DB_NAME, DB_VERSION, (db) => {
				if (!db.objectStoreNames.contains(STORE_BLOBS)) {
					db.createObjectStore(STORE_BLOBS);
				}
				if (!db.objectStoreNames.contains(STORE_META)) {
					db.createObjectStore(STORE_META);
				}
			});
		return this.dbPromise;
	}

	async get(url: string): Promise<ArrayBuffer | undefined> {
		const db = await this.db;
		const [meta, buf] = await Promise.all([readMeta(db), getBlob(db, url)]);
		if (!buf) return undefined;
		// Bump LRU timestamp
		meta.entries[url] = { last: performance.now(), size: buf.byteLength };
		await writeMeta(db, meta);
		console.debug('dataset shard cache hit', url);
		return buf;
	}

	async set(url: string, buf: ArrayBuffer): Promise<void> {
		const db = await this.db;
		const meta = await readMeta(db);
		const size = buf.byteLength;
		const prevSize = meta.entries[url]?.size ?? 0;
		meta.entries[url] = { last: performance.now(), size };
		meta.totalSize += size - prevSize;
		await putBlob(db, url, buf);
		await this.evictIfNeeded(db, meta);
	}

	private async evictIfNeeded(db: IDBDatabase, meta: LruState): Promise<void> {
		try {
			// Keep at most maxBytes
			const { maxBytes } = this.options;
			const keys = Object.keys(meta.entries);
			if (keys.length === 0) return;

			const overBytes = Math.max(0, meta.totalSize - maxBytes);
			if (overBytes === 0) return;

			// Sort by last access ascending
			keys.sort((a, b) => meta.entries[a].last - meta.entries[b].last);

			let bytesToFree = overBytes;
			for (const key of keys) {
				if (bytesToFree <= 0) break;
				const size = meta.entries[key]?.size ?? 0;
				await deleteBlob(db, key);
				delete meta.entries[key];
				meta.totalSize = Math.max(0, meta.totalSize - size);
				if (bytesToFree > 0) bytesToFree = Math.max(0, bytesToFree - size);
			}
		} finally {
			await writeMeta(db, meta);
		}
	}
}

export const globalShardCache = new ShardCache();
