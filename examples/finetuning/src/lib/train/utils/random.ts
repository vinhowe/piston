import { MersenneTwister19937, Random } from 'random-js';

export function parseSeed(seed?: string): number | undefined {
	if (seed === undefined || seed === '') {
		return undefined;
	}

	const parsed = parseInt(seed);
	if (!isNaN(parsed)) {
		return parsed;
	}

	// Simple hash function for string
	let hash = 0;
	for (let i = 0; i < seed.length; i++) {
		const char = seed.charCodeAt(i);
		hash = (hash << 5) - hash + char;
		hash = hash & hash; // Convert to 32bit integer
	}
	return Math.abs(hash);
}

/**
 * Creates a seeded random number generator from a string or undefined seed.
 * If seed is undefined, auto-seeds the generator.
 * If seed is a string that can be parsed as a number, uses the parsed number.
 * Otherwise, uses a simple hash function to convert the string to a number.
 */
export function seededRandom(seed?: number): Random {
	if (seed === undefined) {
		return new Random(MersenneTwister19937.autoSeed());
	}

	return new Random(MersenneTwister19937.seed(seed));
}

export function forkRandom(random: Random): Random {
	return new Random(MersenneTwister19937.seed(random.int32()));
}
