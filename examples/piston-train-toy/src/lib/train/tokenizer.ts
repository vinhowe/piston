/**
 * @fileoverview Simplified Tokenizer implementation adapted from huggingface/transformers.js
 */

import type { ToyTokenizer } from './data/toy/types';

export function decodeSingle(value: number, tokenizer: ToyTokenizer | null): string {
	return tokenizer?.ids?.[value] || `<${value}>`;
}
