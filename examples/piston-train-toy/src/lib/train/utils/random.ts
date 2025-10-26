import { MersenneTwister19937, Random } from 'random-js';

export function seededRandom(): Random {
	return new Random(MersenneTwister19937.autoSeed());
}

export function forkRandom(random: Random): Random {
	return new Random(MersenneTwister19937.seed(random.int32()));
}
