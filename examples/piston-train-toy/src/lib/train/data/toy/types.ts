export interface ToyTokenizer {
	vocab: Record<string, number>;
	ids: Record<number, string>;
	lastToken: number;
	decode(tokens: number[]): string;
}
