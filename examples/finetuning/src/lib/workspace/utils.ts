import { adjectives, animals, uniqueNamesGenerator } from 'unique-names-generator';

export function generateMemorableName(number: number) {
	// by default uniqueNamesGenerator picks one word per dictionary
	const twoWord = uniqueNamesGenerator({
		dictionaries: [adjectives, animals],
		separator: '-',
		style: 'lowerCase',
		length: 2
	});
	return `${twoWord}-${number}`;
}

/**
 * Returns a new array sorted so that any items whose key is found in `priorityKeys`
 * appear first in the specified order, followed by the remaining items sorted by
 * their key alphabetically.
 *
 * Example:
 *   sortWithPriority(['b', 'a', 'c'], (x) => x, ['c']) -> ['c', 'a', 'b']
 */
export function sortWithPriority<T>(
	items: ReadonlyArray<T>,
	getKey: (item: T) => string,
	priorityKeys: ReadonlyArray<string>
): T[] {
	const priorityIndex = new Map<string, number>();
	for (let i = 0; i < priorityKeys.length; i++) {
		priorityIndex.set(priorityKeys[i], i);
	}
	return [...items].sort((a, b) => {
		const ka = getKey(a);
		const kb = getKey(b);
		const ia = priorityIndex.has(ka) ? (priorityIndex.get(ka) as number) : Number.POSITIVE_INFINITY;
		const ib = priorityIndex.has(kb) ? (priorityIndex.get(kb) as number) : Number.POSITIVE_INFINITY;
		if (ia !== ib) return ia - ib;
		return ka.localeCompare(kb);
	});
}
