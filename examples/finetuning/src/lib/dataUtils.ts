export function openDb(
	dbName: string,
	dbVersion: number,
	onupgradeneeded: (db: IDBDatabase) => void
): Promise<IDBDatabase> {
	return new Promise((resolve, reject) => {
		const req = indexedDB.open(dbName, dbVersion);
		req.onupgradeneeded = () => onupgradeneeded(req.result);
		req.onsuccess = () => resolve(req.result);
		req.onerror = () => reject(req.error);
		req.onblocked = () => reject(new Error('IndexedDB upgrade blocked'));
	});
}

export function promisify<T>(req: IDBRequest<T>): Promise<T> {
	return new Promise((resolve, reject) => {
		req.onsuccess = () => resolve(req.result);
		req.onerror = () => reject(req.error);
	});
}

export function txRequest<T>(
	db: IDBDatabase,
	storeName: string,
	mode: IDBTransactionMode,
	op: (store: IDBObjectStore) => IDBRequest<T>
): Promise<T> {
	const tx = db.transaction(storeName, mode);
	const store = tx.objectStore(storeName);
	return promisify(op(store));
}
