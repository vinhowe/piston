/**
 * @fileoverview Simplified Tokenizer implementation adapted from huggingface/transformers.js
 */

import { PUBLIC_DATA_URL } from '$env/static/public';
import { Template } from '@huggingface/jinja';
import { int32, Tensor, tensor } from '@piston-ml/piston-web';

import type { ToyTokenizer } from './data/toy/types';

/* eslint-disable @typescript-eslint/no-unsafe-declaration-merging */
abstract class Callable<Args extends unknown[] = unknown[], Return = unknown> {
	/**
	 * Creates a new instance of the Callable class.
	 */
	constructor() {
		/**
		 * Creates a closure that delegates to a private method 'call' with the given arguments.
		 * @param args Zero or more arguments to pass to the 'call' method.
		 * @returns The result of calling the 'call' method.
		 */
		const closure = ((...args: Args) => {
			return (closure as unknown as { call: (...args: Args) => Return }).call(...args);
		}) as unknown as (...args: Args) => Return;
		return Object.setPrototypeOf(closure, new.target.prototype) as unknown as this &
			((...args: Args) => Return);
	}

	/**
	 * This method should be implemented in subclasses to provide the
	 * functionality of the callable object.
	 *
	 * @param args Zero or more arguments to pass to the 'call' method.
	 * @throws {Error} If the subclass does not implement the `call` method.
	 */
	protected abstract call(..._args: Args): Return;
}
interface Callable<Args extends unknown[] = unknown[], Return = unknown> {
	(...args: Args): Return;
}

// Discriminated config helpers
type WithType<TType extends string> = { type: TType };

// Normalizer configs
type NormalizerSequenceConfig = WithType<'Sequence'> & { normalizers: NormalizerConfig[] };
type NFCConfig = WithType<'NFC'>;
type NFDConfig = WithType<'NFD'>;
type NFKCConfig = WithType<'NFKC'>;
type NFKDConfig = WithType<'NFKD'>;
type StripConfig = WithType<'Strip'> & { stripLeft?: boolean; stripRight?: boolean };
type LowercaseConfig = WithType<'Lowercase'>;
type PrependConfig = WithType<'Prepend'> & { prepend: string };
type NormalizerConfig =
	| NormalizerSequenceConfig
	| NFCConfig
	| NFDConfig
	| NFKCConfig
	| NFKDConfig
	| StripConfig
	| LowercaseConfig
	| PrependConfig;

// PreTokenizer configs and options
type PreTokenizeOptions = { sectionIndex?: number };
type PreTokenizerSequenceConfig = WithType<'Sequence'> & { pretokenizers: PreTokenizerConfig[] };
type WhitespacePreTokenizerConfig = WithType<'Whitespace'>;
type WhitespaceSplitConfig = WithType<'WhitespaceSplit'>;
type MetaspacePreTokenizerConfig = WithType<'Metaspace'> & {
	addPrefixSpace: boolean;
	replacement: string;
	strRep?: string;
	prependScheme?: 'first' | 'never' | 'always';
};
type ByteLevelPreTokenizerConfig = WithType<'ByteLevel'> & {
	addPrefixSpace: boolean;
	trimOffsets: boolean;
	useRegex?: boolean;
};
type PreTokenizerConfig =
	| PreTokenizerSequenceConfig
	| WhitespacePreTokenizerConfig
	| WhitespaceSplitConfig
	| MetaspacePreTokenizerConfig
	| ByteLevelPreTokenizerConfig;

// PostProcessor configs and options
type PostProcessorOptions = { addSpecialTokens?: boolean };
type PostProcessorResult = { tokens: string[]; tokenTypeIds?: number[] };
type PostProcessorSequenceConfig = WithType<'Sequence'> & { processors: PostProcessorConfig[] };
type ByteLevelPostProcessorConfig = WithType<'ByteLevel'>;
type PostProcessorConfig = PostProcessorSequenceConfig | ByteLevelPostProcessorConfig;

// Decoder configs
type ByteLevelDecoderConfig = WithType<'ByteLevel'> & { trimOffsets?: boolean };
type ByteFallbackConfig = WithType<'ByteFallback'>;
type FuseDecoderConfig = WithType<'Fuse'>;
type StripDecoderConfig = WithType<'Strip'> & { content: string; start: number; stop: number };
type DecoderSequenceConfig = WithType<'Sequence'> & { decoders: DecoderConfig[] };
type BPEDecoderConfig = WithType<'BPEDecoder'> & { suffix: string };
type DecoderConfig =
	| ByteLevelDecoderConfig
	| ByteFallbackConfig
	| FuseDecoderConfig
	| StripDecoderConfig
	| DecoderSequenceConfig
	| BPEDecoderConfig;

// Model configs
export interface TokenizerModelConfig {
	fuseUnk: boolean;
	byteFallback: boolean;
	ignoreMerges: boolean;
}

type BPEConfig = WithType<'BPE'> &
	TokenizerModelConfig & {
		vocab: Record<string, number>;
		merges: string[] | [string, string][];
		unkToken: string;
		endOfWordSuffix?: string;
		continuingSubwordSuffix?: string | null;
	};

type TokenizerModelFactoryConfig = BPEConfig; // Extend when additional models are added

// Tokenizer JSON and runtime config
interface TokenizerJSON {
	normalizer: NormalizerConfig | null;
	preTokenizer: PreTokenizerConfig | null;
	model: TokenizerModelFactoryConfig;
	postProcessor: PostProcessorConfig | null;
	decoder: DecoderConfig | null;
	addedTokens: AddedTokenConfig[];
}

interface TokenizerConfig {
	[key: string]: unknown;
	additionalSpecialTokens?: string[];
	modelMaxLength: number;
	removeSpace: boolean;
	cleanUpTokenizationSpaces?: boolean;
	paddingSide?: 'left' | 'right';
	addBosToken?: boolean;
	addEosToken?: boolean;
	chatTemplate?: null | Array<{ name: string; template: string }> | Record<string, string>;
}

const TOKENIZER_URL = PUBLIC_DATA_URL + 'tokenizer';

/**
 * Loads a tokenizer from the specified path.
 * @param tokenizerName The path to the tokenizer directory.
 * @returns A promise that resolves with tokenizer JSON and config.
 */
async function loadTokenizer(tokenizerName: string): Promise<[TokenizerJSON, TokenizerConfig]> {
	return Promise.all([
		fetchJSON<TokenizerJSON>(TOKENIZER_URL, `${tokenizerName}/tokenizer.json`).then(
			camelCaseKeysDeep
		),
		fetchJSON<TokenizerConfig>(TOKENIZER_URL, `${tokenizerName}/tokenizer_config.json`).then(
			camelCaseKeysDeep
		)
	]);
}

function isPlainObject(value: unknown): value is Record<string, unknown> {
	return (
		value !== null && typeof value === 'object' && Object.getPrototypeOf(value) === Object.prototype
	);
}

function toCamelKey(key: string): string {
	return key.includes('_')
		? key.replace(/_+([a-zA-Z0-9])/g, (_m, c: string) => c.toUpperCase())
		: key;
}

function camelCaseKeysDeep<T>(input: T): T {
	if (Array.isArray(input)) {
		return input.map((item) => camelCaseKeysDeep(item)) as unknown as T;
	}
	if (isPlainObject(input)) {
		const obj = input as Record<string, unknown>;
		const out: Record<string, unknown> = Object.create(null);
		for (const [key, value] of Object.entries(obj)) {
			const transformed = camelCaseKeysDeep(value);
			// Preserve original snake_case for compatibility
			out[key] = transformed;
			const camelKey = toCamelKey(key);
			if (camelKey !== key && !(camelKey in out)) {
				out[camelKey] = transformed;
			}
		}
		return out as unknown as T;
	}
	return input;
}

// Minimal fetch wrapper used here; replace with project-util if available
async function fetchJSON<T>(basePath: string, fileName: string): Promise<T> {
	const url = `${basePath.replace(/\/$/, '')}/${fileName}`;
	const res = await fetch(url);
	if (!res.ok) throw new Error(`Failed to load ${fileName} from ${url}`);
	return res.json() as Promise<T>;
}

/**
 * Helper function to convert an Object to a Map
 * @param obj The object to convert.
 * @returns The map.
 */
function objectToMap<T>(obj: Record<string, T>): Map<string, T> {
	return new Map(Object.entries(obj));
}

/**
 * Helper function to fuse consecutive unknown tokens.
 * @param arr The list of input tokens
 * @param tokensToIds The mapping from tokens to token ids.
 * @param unkTokenId The value to fuse on.
 */
function fuseUnk(arr: string[], tokensToIds: Map<string, number>, unkTokenId: number): string[] {
	const fused = [];
	let i = 0;
	while (i < arr.length) {
		fused.push(arr[i]);
		if ((tokensToIds.get(arr[i]) ?? unkTokenId) !== unkTokenId) {
			++i;
			continue;
		}

		while (++i < arr.length && (tokensToIds.get(arr[i]) ?? unkTokenId) === unkTokenId) {
			if (tokensToIds.get(fused[fused.length - 1]) !== unkTokenId) {
				fused[fused.length - 1] += arr[i];
			}
		}
	}

	return fused;
}

/**
 * Split a string on whitespace.
 * @param text The text to split.
 * @returns The split string.
 */
function whitespaceSplit(text: string): string[] {
	return text.match(/\S+/g) || [];
}

/**
 * Represent a token added by the user on top of the existing Model vocabulary.
 * AddedToken can be configured to specify the behavior they should have in various situations like:
 *   - Whether they should only match single words
 *   - Whether to include any whitespace on its left or right
 */
interface AddedTokenConfig {
	content: string;
	id: number;
	singleWord?: boolean;
	lstrip?: boolean;
	rstrip?: boolean;
	normalized?: boolean;
	special?: boolean;
}
class AddedToken {
	content: string;
	id: number;
	singleWord: boolean;
	lstrip: boolean;
	rstrip: boolean;
	special: boolean;
	normalized: boolean | null;
	/**
	 * Creates a new instance of AddedToken.
	 * @param config Added token configuration object.
	 * @param config.content The content of the added token.
	 * @param config.id The id of the added token.
	 * @param config.singleWord Whether this token must be a single word or can break words.
	 * @param config.lstrip Whether this token should strip whitespaces on its left.
	 * @param config.rstrip Whether this token should strip whitespaces on its right.
	 * @param config.normalized Whether this token should be normalized.
	 * @param config.special Whether this token is special.
	 */
	constructor(config: AddedTokenConfig) {
		this.content = config.content;
		this.id = config.id;
		this.singleWord = config.singleWord ?? false;
		this.lstrip = config.lstrip ?? false;
		this.rstrip = config.rstrip ?? false;
		this.special = config.special ?? false;
		this.normalized = config.normalized ?? null;
	}
}

export interface TokenizerModelConfig {
	fuseUnk: boolean;
	byteFallback: boolean;
	ignoreMerges: boolean;
}

/**
 * Abstract base class for tokenizer models.
 */
export class TokenizerModel extends Callable<[string[]], string[]> {
	config: TokenizerModelConfig;
	vocab: string[];
	tokensToIds: Map<string, number>;
	unkTokenId?: number;
	unkToken?: string;
	endOfWordSuffix?: string;
	fuseUnk: boolean;
	/**
	 * Creates a new instance of TokenizerModel.
	 * @param config The configuration object for the TokenizerModel.
	 */
	constructor(config: TokenizerModelConfig) {
		super();
		this.config = config;

		this.vocab = [];

		this.tokensToIds = new Map();

		this.unkTokenId = undefined;
		this.unkToken = undefined;
		this.endOfWordSuffix = undefined;

		this.fuseUnk = this.config.fuseUnk ?? false;
	}

	/**
	 * Instantiates a new TokenizerModel instance based on the configuration object provided.
	 * @param config The configuration object for the TokenizerModel.
	 * @param _args Optional arguments to pass to the specific TokenizerModel constructor.
	 * @returns A new instance of a TokenizerModel.
	 * @throws Will throw an error if the TokenizerModel type in the config is not recognized.
	 */
	static fromConfig(config: TokenizerModelFactoryConfig, ..._args: unknown[]): TokenizerModel {
		switch (config.type) {
			case 'BPE':
			default:
				return new BPE(config);
		}
	}

	/**
	 * Internal function to call the TokenizerModel instance.
	 * @param tokens The tokens to encode.
	 * @returns The encoded tokens.
	 */
	protected call(...[tokens]: [string[]]): string[] {
		tokens = this.encode(tokens);
		if (this.fuseUnk) {
			// Fuse unknown tokens
			tokens = fuseUnk(tokens, this.tokensToIds, this.unkTokenId as number);
		}
		return tokens;
	}

	/**
	 * Encodes a list of tokens into a list of token IDs.
	 * @param tokens The tokens to encode.
	 * @returns The encoded tokens.
	 * @throws Will throw an error if not implemented in a subclass.
	 */
	encode(_tokens: string[]): string[] {
		throw Error('encode should be implemented in subclass.');
	}

	/**
	 * Converts a list of tokens into a list of token IDs.
	 * @param tokens The tokens to convert.
	 * @returns The converted token IDs.
	 */
	convertTokensToIds(tokens: string[]): number[] {
		return tokens.map((t) => this.tokensToIds.get(t) ?? (this.unkTokenId as number));
	}

	/**
	 * Converts a list of token IDs into a list of tokens.
	 * @param ids The token IDs to convert.
	 * @returns The converted tokens.
	 */
	convertIdsToTokens(ids: number[] | bigint[]): string[] {
		return ids.map((i) => this.vocab[Number(i)] ?? (this.unkToken as string));
	}
}

/**
 * Returns list of utf-8 byte and a mapping to unicode strings.
 * Specifically avoids mapping to whitespace/control characters the BPE code barfs on.
 * @returns Object with utf-8 byte keys and unicode string values.
 */
const BYTES_TO_UNICODE = (() => {
	// Returns list of utf-8 byte and a mapping to unicode strings.
	// We specifically avoids mapping to whitespace/control characters the bpe code barfs on.

	const bs = [
		...Array.from(
			{ length: '~'.charCodeAt(0) - '!'.charCodeAt(0) + 1 },
			(_, i) => i + '!'.charCodeAt(0)
		),
		...Array.from(
			{ length: '¬'.charCodeAt(0) - '¡'.charCodeAt(0) + 1 },
			(_, i) => i + '¡'.charCodeAt(0)
		),
		...Array.from(
			{ length: 'ÿ'.charCodeAt(0) - '®'.charCodeAt(0) + 1 },
			(_, i) => i + '®'.charCodeAt(0)
		)
	];
	const cs = bs.slice();
	let n = 0;
	for (let b = 0; b < 256; ++b) {
		if (!bs.includes(b)) {
			bs.push(b);
			cs.push(256 + n);
			n += 1;
		}
	}
	const ccs = cs.map((n) => String.fromCharCode(n));
	return Object.fromEntries(bs.map((b, i) => [b, ccs[i]]));
})();

const UNICODE_TO_BYTES = Object.fromEntries(
	Object.entries(BYTES_TO_UNICODE).map(([key, value]) => [value, key])
);

interface BPENode {
	token: string;
	bias: number;
	score?: number;
	prev?: BPENode;
	next?: BPENode;
}

/**
 * BPE class for encoding text into Byte-Pair-Encoding (BPE) tokens.
 */
class BPE extends TokenizerModel {
	merges!: [string, string][];
	bpeRanks!: Map<string, number>;
	continuingSubwordSuffix!: string | null;
	byteFallback!: boolean;
	textEncoder!: TextEncoder;
	ignoreMerges!: boolean;
	maxLengthToCache!: number;
	cacheCapacity!: number;
	cache!: LRUCache<string, string[]>;

	constructor(config: BPEConfig) {
		super(config);
		this.tokensToIds = objectToMap(config.vocab);
		this.unkTokenId = this.tokensToIds.get(config.unkToken) as number;
		this.unkToken = config.unkToken as string;
		this.vocab = new Array(this.tokensToIds.size);
		for (const [key, value] of this.tokensToIds) {
			this.vocab[value] = key;
		}
		const useNewMergeFormat = Array.isArray(config.merges[0]);
		this.merges = useNewMergeFormat
			? (config.merges as [string, string][])
			: (config.merges as string[]).map((x) => x.split(' ', 2) as [string, string]);
		this.bpeRanks = new Map(this.merges.map((x, i) => [JSON.stringify(x), i])) as Map<
			string,
			number
		>;
		this.endOfWordSuffix = config.endOfWordSuffix as string | undefined;
		this.continuingSubwordSuffix = (config.continuingSubwordSuffix ?? null) as string | null;
		this.byteFallback = (this.config.byteFallback ?? false) as boolean;
		if (this.byteFallback) {
			this.textEncoder = new TextEncoder();
		}
		this.ignoreMerges = (this.config.ignoreMerges ?? false) as boolean;
		this.maxLengthToCache = 256;
		this.cacheCapacity = 10000;
		this.cache = new LRUCache(this.cacheCapacity);
	}
	clearCache() {
		this.cache.clear();
	}
	bpe(token: string): string[] {
		if (token.length === 0) {
			return [];
		}
		const cached = this.cache.get(token);
		if (cached !== undefined) {
			return cached;
		}
		const word = Array.from(token);
		if (this.endOfWordSuffix) {
			word[word.length - 1] += this.endOfWordSuffix;
		}
		let result: string[] = [];
		if (word.length > 1) {
			const queue = new PriorityQueue<BPENode>((a, b) => (a.score as number) < (b.score as number));
			let startingNode: BPENode = {
				token: word[0],
				bias: 0,
				prev: undefined,
				next: undefined
			};
			let previousNode = startingNode;
			for (let i = 1; i < word.length; ++i) {
				const currentNode: BPENode = {
					bias: i / word.length,
					token: word[i],
					prev: previousNode,
					next: undefined
				};
				previousNode.next = currentNode;
				this.addNode(queue, previousNode);
				previousNode = currentNode;
			}
			while (!queue.isEmpty()) {
				const node = queue.pop() as BPENode & {
					deleted?: boolean;
					prev?: BPENode & { deleted?: boolean };
					next?: BPENode & { deleted?: boolean };
				};
				if (node.deleted || !node.next || node.next.deleted) continue;
				node.deleted = true;
				node.next.deleted = true;
				if (node.prev) {
					const newPreviousNode = { ...(node.prev as BPENode) } as BPENode;
					node.prev.deleted = true;
					node.prev = newPreviousNode;
					if (newPreviousNode.prev) {
						(newPreviousNode.prev as BPENode).next = newPreviousNode;
					} else {
						startingNode = newPreviousNode;
					}
				}
				const merged: BPENode = {
					token: node.token + (node.next as BPENode).token,
					bias: node.bias,
					prev: node.prev,
					next: (node.next as BPENode).next
				};
				if (merged.prev) {
					(merged.prev as BPENode).next = merged;
					this.addNode(queue, merged.prev as BPENode);
				} else {
					startingNode = merged;
				}
				if (merged.next) {
					(merged.next as BPENode).prev = merged;
					this.addNode(queue, merged);
				}
			}
			for (
				let currentNode: BPENode | undefined = startingNode;
				currentNode !== undefined;
				currentNode = currentNode.next
			) {
				result.push(currentNode.token);
			}
		} else {
			result = word;
		}
		if (this.continuingSubwordSuffix) {
			for (let i = 0; i < result.length - 1; ++i) {
				result[i] += this.continuingSubwordSuffix;
			}
		}
		if (token.length < this.maxLengthToCache) {
			this.cache.put(token, result);
		}
		return result;
	}
	private addNode(queue: PriorityQueue<BPENode>, node: BPENode) {
		const rank = this.bpeRanks.get(JSON.stringify([node.token, (node.next as BPENode).token]));
		if (rank !== undefined) {
			node.score = rank + node.bias;
			queue.push(node);
		}
	}
	encode(tokens: string[]): string[] {
		const outputTokens: string[] = [];
		for (const token of tokens) {
			if (this.ignoreMerges && this.tokensToIds.has(token)) {
				outputTokens.push(token);
				continue;
			}
			const bpeTokenList = this.bpe(token);
			for (const t of bpeTokenList) {
				if (this.tokensToIds.has(t)) {
					outputTokens.push(t);
				} else if (this.byteFallback) {
					const byteTokens = Array.from(this.textEncoder.encode(t)).map(
						(x) => `<0x${x.toString(16).toUpperCase().padStart(2, '0')}>`
					);
					if (byteTokens.every((x) => this.tokensToIds.has(x))) {
						outputTokens.push(...byteTokens);
					} else {
						outputTokens.push(this.unkToken as string);
					}
				} else {
					outputTokens.push(this.unkToken as string);
				}
			}
		}
		return outputTokens;
	}
}

/**
 * A base class for text normalization.
 */
abstract class Normalizer<TConfig = unknown> extends Callable<[string], string> {
	config: TConfig;
	/**
	 * @param config The configuration object for the normalizer.
	 */
	constructor(config: TConfig) {
		super();
		this.config = config;
	}
	static fromConfig<TConfig extends NormalizerConfig>(config: TConfig): Normalizer<unknown> {
		switch (config.type) {
			case 'Sequence':
				return new NormalizerSequence(config);
			case 'NFC':
				return new NFC(config);
			case 'NFD':
				return new NFD(config);
			case 'NFKC':
				return new NFKC(config);
			case 'NFKD':
				return new NFKD(config);
			case 'Strip':
				return new StripNormalizer(config);
			case 'Lowercase':
				return new Lowercase(config);
			case 'Prepend':
				return new Prepend(config);
		}
	}

	normalize(_text: string): string {
		throw Error('normalize should be implemented in subclass.');
	}

	protected call(...[text]: [string]): string {
		return this.normalize(text);
	}
}

/**
 * A normalizer that applies Unicode normalization to the input text.
 */
abstract class UnicodeNormalizer extends Normalizer {
	form: 'NFC' | 'NFD' | 'NFKC' | 'NFKD' | undefined = undefined;

	/**
	 * Normalize the input text by applying Unicode normalization.
	 * @param text The input text to be normalized.
	 * @returns The normalized text.
	 */
	normalize(text: string) {
		text = text.normalize(this.form as 'NFC');
		return text;
	}
}

/**
 * A normalizer that applies Unicode normalization form C (NFC) to the input text.
 * Canonical Decomposition, followed by Canonical Composition.
 */
class NFC extends UnicodeNormalizer {
	form = 'NFC' as const;
}

/**
 * A normalizer that applies Unicode normalization form D (NFD) to the input text.
 * Canonical Decomposition.
 */
class NFD extends UnicodeNormalizer {
	form = 'NFD' as const;
}

/**
 * A normalizer that applies Unicode normalization form KC (NFKC) to the input text.
 * Compatibility Decomposition, followed by Canonical Composition.
 */
class NFKC extends UnicodeNormalizer {
	form = 'NFKC' as const;
}

/**
 * A normalizer that applies Unicode normalization form KD (NFKD) to the input text.
 * Compatibility Decomposition.
 */
class NFKD extends UnicodeNormalizer {
	form = 'NFKD' as const;
}

/**
 * A normalizer that strips leading and/or trailing whitespace from the input text.
 */
class StripNormalizer extends Normalizer<StripConfig> {
	/**
	 * Strip leading and/or trailing whitespace from the input text.
	 * @param text The input text.
	 * @returns The normalized text.
	 */
	normalize(text: string) {
		const cfg = this.config;
		if (cfg.stripLeft && cfg.stripRight) {
			// Fast path to avoid an extra trim call
			text = text.trim();
		} else {
			if (cfg.stripLeft) {
				text = text.trimStart();
			}
			if (cfg.stripRight) {
				text = text.trimEnd();
			}
		}
		return text;
	}
}

/**
 * A Normalizer that lowercases the input string.
 */
class Lowercase extends Normalizer {
	/**
	 * Lowercases the input string.
	 * @param text The text to normalize.
	 * @returns The normalized text.
	 */
	normalize(text: string) {
		text = text.toLowerCase();
		return text;
	}
}

/**
 * A Normalizer that prepends a string to the input string.
 */
class Prepend extends Normalizer<PrependConfig> {
	/**
	 * Prepends the input string.
	 * @param text The text to normalize.
	 * @returns The normalized text.
	 */
	normalize(text: string) {
		const cfg = this.config;
		text = cfg.prepend + text;
		return text;
	}
}

/**
 * A Normalizer that applies a sequence of Normalizers.
 */

class NormalizerSequence extends Normalizer<NormalizerSequenceConfig> {
	normalizers: Normalizer<unknown>[];

	constructor(config: NormalizerSequenceConfig) {
		super(config);
		this.normalizers = config.normalizers.map((x) => Normalizer.fromConfig(x));
	}
	/**
	 * Apply a sequence of Normalizers to the input text.
	 * @param text The text to normalize.
	 * @returns The normalized text.
	 */
	normalize(text: string) {
		return this.normalizers.reduce((t, normalizer) => {
			return normalizer.normalize(t);
		}, text);
	}
}

/**
 * A callable class representing a pre-tokenizer used in tokenization. Subclasses
 * should implement the `preTokenizeText` method to define the specific pre-tokenization logic.
 */
abstract class PreTokenizer extends Callable<
	[string | string[], PreTokenizeOptions | undefined],
	string[]
> {
	/**
	 * Factory method that returns an instance of a subclass of `PreTokenizer` based on the provided configuration.
	 *
	 * @static
	 * @param config A configuration object for the pre-tokenizer.
	 * @returns An instance of a subclass of `PreTokenizer`.
	 * @throws If the provided configuration object does not correspond to any known pre-tokenizer.
	 */
	static fromConfig(config: PreTokenizerConfig): PreTokenizer {
		switch (config.type) {
			case 'Sequence':
				return new PreTokenizerSequence(config);
			case 'Whitespace':
				return new WhitespacePreTokenizer();
			case 'WhitespaceSplit':
				return new WhitespaceSplit();
			case 'Metaspace':
				return new MetaspacePreTokenizer(config);
			case 'ByteLevel':
				return new ByteLevelPreTokenizer(config);
			default:
				throw new Error('Unknown PreTokenizer type');
		}
	}

	/**
	 * Method that should be implemented by subclasses to define the specific pre-tokenization logic.
	 *
	 * @param text The text to pre-tokenize.
	 * @param options Additional options for the pre-tokenization logic.
	 * @returns The pre-tokenized text.
	 * @throws {Error} If the method is not implemented in the subclass.
	 */
	abstract preTokenizeText(text: string, options?: PreTokenizeOptions): string[];

	/**
	 * Tokenizes the given text into pre-tokens.
	 * @param text The text or array of texts to pre-tokenize.
	 * @param options Additional options for the pre-tokenization logic.
	 * @returns An array of pre-tokens.
	 */
	preTokenize(text: string | string[], options?: PreTokenizeOptions): string[] {
		return (
			Array.isArray(text)
				? (text as string[]).map((x) => this.preTokenizeText(x, options))
				: this.preTokenizeText(text as string, options)
		).flat();
	}

	/**
	 * Alias for {@link PreTokenizer#preTokenize}.
	 * @param text The text or array of texts to pre-tokenize.
	 * @param options Additional options for the pre-tokenization logic.
	 * @returns An array of pre-tokens.
	 */
	protected call(
		...[text, options]: [string | string[], PreTokenizeOptions | undefined]
	): string[] {
		return this.preTokenize(text, options);
	}
}

/**
 * A pre-tokenizer that splits text into Byte-Pair-Encoding (BPE) subwords.
 * @extends PreTokenizer
 */

class ByteLevelPreTokenizer extends PreTokenizer {
	config: ByteLevelPreTokenizerConfig;
	addPrefixSpace!: boolean;
	trimOffsets!: boolean;
	useRegex!: boolean;
	pattern!: RegExp;
	byteEncoder!: Record<number, string>;
	textEncoder!: TextEncoder;
	constructor(config: ByteLevelPreTokenizerConfig) {
		super();
		this.config = config;
		this.addPrefixSpace = this.config.addPrefixSpace;
		this.trimOffsets = this.config.trimOffsets;
		this.useRegex = this.config.useRegex ?? true;
		this.pattern = /'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+/gu;
		this.byteEncoder = BYTES_TO_UNICODE as Record<number, string>;
		this.textEncoder = new TextEncoder();
	}
	preTokenizeText(text: string, _options?: PreTokenizeOptions): string[] {
		if (this.addPrefixSpace && !text.startsWith(' ')) {
			text = ' ' + text;
		}
		const tokens = this.useRegex ? text.match(this.pattern) || [] : [text];
		return tokens.map((token) =>
			Array.from(this.textEncoder.encode(token), (byte) => this.byteEncoder[byte]).join('')
		);
	}
}

type PostProcessorArgs = [string[], (string[] | null | undefined)?, PostProcessorOptions?];
abstract class PostProcessor extends Callable<PostProcessorArgs, PostProcessorResult> {
	config: PostProcessorConfig;
	constructor(config: PostProcessorConfig) {
		super();
		this.config = config;
	}
	static fromConfig(config: PostProcessorConfig): PostProcessor {
		switch (config.type) {
			case 'ByteLevel':
				return new ByteLevelPostProcessor(config);
			case 'Sequence':
				return new PostProcessorSequence(config);
			default:
				throw new Error('Unknown PostProcessor type');
		}
	}

	abstract postProcess(
		tokens: string[],
		...args: [string[] | null | undefined, PostProcessorOptions?]
	): PostProcessorResult;

	protected call(
		...[tokens, ...args]: [string[], (string[] | null | undefined)?, PostProcessorOptions?]
	): PostProcessorResult {
		return this.postProcess(tokens, ...args);
	}
}

/**
 * A PostProcessor that returns the given tokens as is.
 */
class ByteLevelPostProcessor extends PostProcessor {
	postProcess(tokens: string[], tokensPair: string[] | null = null): { tokens: string[] } {
		if (tokensPair) {
			tokens = mergeArrays(tokens, tokensPair);
		}
		return { tokens };
	}
}

/**
 * A post-processor that applies multiple post-processors in sequence.
 */
class PostProcessorSequence extends PostProcessor {
	processors: PostProcessor[];
	constructor(config: PostProcessorSequenceConfig) {
		super(config);
		this.processors = config.processors.map((x) => PostProcessor.fromConfig(x));
	}
	postProcess(
		tokens: string[],
		tokensPair: string[] | null = null,
		options: PostProcessorOptions = {}
	): { tokens: string[]; tokenTypeIds?: number[] } {
		let tokenTypeIds: number[] | undefined;
		for (const processor of this.processors) {
			if (processor instanceof ByteLevelPostProcessor) {
				const output = processor.postProcess(tokens);
				tokens = output.tokens;
				if (tokensPair) {
					const pairOutput = processor.postProcess(tokensPair);
					tokensPair = pairOutput.tokens;
				}
			} else {
				const output = processor.postProcess(tokens, tokensPair ?? null, options);
				tokens = output.tokens;
				if (output.tokenTypeIds) {
					tokenTypeIds = output.tokenTypeIds;
				}
			}
		}
		return { tokens, tokenTypeIds: tokenTypeIds };
	}
}

/**
 * The base class for token decoders.
 */
abstract class Decoder<TConfig extends DecoderConfig = DecoderConfig> extends Callable<
	[string[]],
	string
> {
	config: TConfig;
	addedTokens: AddedToken[];
	endOfWordSuffix?: string;
	constructor(config: TConfig) {
		super();
		this.config = config;
		this.addedTokens = [];
		this.endOfWordSuffix = undefined;
	}
	static fromConfig(config: DecoderConfig): Decoder<DecoderConfig> {
		switch (config.type) {
			case 'ByteLevel':
				return new ByteLevelDecoder(config);
			case 'ByteFallback':
				return new ByteFallback(config);
			case 'Fuse':
				return new FuseDecoder(config);
			case 'Strip':
				return new StripDecoder(config);
			case 'Sequence':
				return new DecoderSequence(config);
			case 'BPEDecoder':
				return new BPEDecoder(config);
			default:
				throw new Error('Unknown Decoder type');
		}
	}
	protected call(...[tokens]: [string[]]): string {
		return this.decode(tokens);
	}
	decode(tokens: string[]): string {
		return this.decodeChain(tokens).join('');
	}
	abstract decodeChain(tokens: string[]): string[];
}

class ByteFallback extends Decoder<ByteFallbackConfig> {
	textDecoder!: TextDecoder;
	constructor(config: ByteFallbackConfig) {
		super(config);
		this.textDecoder = new TextDecoder();
	}
	decodeChain(tokens: string[]): string[] {
		const newTokens: string[] = [];
		let previousByteTokens: number[] = [];
		for (const token of tokens) {
			let bytes: number | null = null;
			if (token.length === 6 && token.startsWith('<0x') && token.endsWith('>')) {
				const byte = parseInt(token.slice(3, 5), 16);
				if (!isNaN(byte)) {
					bytes = byte;
				}
			}
			if (bytes !== null) {
				previousByteTokens.push(bytes);
			} else {
				if (previousByteTokens.length > 0) {
					const string = this.textDecoder.decode(Uint8Array.from(previousByteTokens));
					newTokens.push(string);
					previousByteTokens = [];
				}
				newTokens.push(token);
			}
		}
		if (previousByteTokens.length > 0) {
			const string = this.textDecoder.decode(Uint8Array.from(previousByteTokens));
			newTokens.push(string);
			previousByteTokens = [];
		}
		return newTokens;
	}
}

/**
 * Fuse simply fuses all tokens into one big string.
 * It's usually the last decoding step anyway, but this decoder
 * exists incase some decoders need to happen after that step
 */
class FuseDecoder extends Decoder<FuseDecoderConfig> {
	/** @type {Decoder['decodeChain']} */
	decodeChain(tokens: string[]): string[] {
		return [tokens.join('')];
	}
}

class StripDecoder extends Decoder<StripDecoderConfig> {
	content!: string;
	start!: number;
	stop!: number;
	constructor(config: StripDecoderConfig) {
		super(config);
		const cfg = this.config;
		this.content = cfg.content;
		this.start = cfg.start;
		this.stop = cfg.stop;
	}
	/** @type {Decoder['decodeChain']} */
	decodeChain(tokens: string[]): string[] {
		return tokens.map((token) => {
			let startCut = 0;
			for (let i = 0; i < this.start; ++i) {
				if (token[i] === this.content) {
					startCut = i + 1;
					continue;
				} else {
					break;
				}
			}
			let stopCut = token.length;
			for (let i = 0; i < this.stop; ++i) {
				const index = token.length - i - 1;
				if (token[index] === this.content) {
					stopCut = index;
					continue;
				} else {
					break;
				}
			}
			return token.slice(startCut, stopCut);
		});
	}
}

/**
 * Byte-level decoder for tokenization output. Inherits from the `Decoder` class.
 * @extends Decoder
 */
class ByteLevelDecoder extends Decoder<ByteLevelDecoderConfig> {
	byteDecoder!: Record<string, number>;
	textDecoder!: TextDecoder;
	constructor(config: ByteLevelDecoderConfig) {
		super(config);
		this.byteDecoder = UNICODE_TO_BYTES as unknown as Record<string, number>;
		this.textDecoder = new TextDecoder('utf-8', { fatal: false, ignoreBOM: true });
		this.endOfWordSuffix = undefined;
	}
	convertTokensToString(tokens: string[]): string {
		const text = tokens.join('');
		const byteArray = new Uint8Array([...text].map((c) => this.byteDecoder[c]));
		const decodedText = this.textDecoder.decode(byteArray);
		return decodedText;
	}
	/** @type {Decoder['decodeChain']} */
	decodeChain(tokens: string[]): string[] {
		const subTexts: string[] = [];
		let currentSubText: string[] = [];
		for (const token of tokens) {
			if (this.addedTokens.find((x) => x.content === token) !== undefined) {
				if (currentSubText.length > 0) {
					subTexts.push(this.convertTokensToString(currentSubText));
					currentSubText = [];
				}
				subTexts.push(token);
			} else {
				currentSubText.push(token);
			}
		}
		if (currentSubText.length > 0) {
			subTexts.push(this.convertTokensToString(currentSubText));
		}
		return subTexts;
	}
}

/**
 * Apply a sequence of decoders.
 * @extends Decoder
 */
class DecoderSequence extends Decoder<DecoderSequenceConfig> {
	decoders!: Decoder[];
	constructor(config: DecoderSequenceConfig) {
		super(config);
		this.decoders = config.decoders.map((x) => Decoder.fromConfig(x));
	}
	/** @type {Decoder['decodeChain']} */
	decodeChain(tokens: string[]): string[] {
		return this.decoders.reduce((toks: string[], decoder: Decoder) => {
			return decoder.decodeChain(toks);
		}, tokens);
	}
}

class BPEDecoder extends Decoder<BPEDecoderConfig> {
	suffix!: string;
	constructor(config: BPEDecoderConfig) {
		super(config);
		const cfg = this.config;
		this.suffix = cfg.suffix;
	}
	/** @type {Decoder['decodeChain']} */
	decodeChain(tokens: string[]): string[] {
		return tokens.map((token, i) => {
			return token.replaceAll(this.suffix, i === tokens.length - 1 ? '' : ' ');
		});
	}
}

/**
 * This PreTokenizer replaces spaces with the given replacement character, adds a prefix space if requested,
 * and returns a list of tokens.
 * @extends PreTokenizer
 */
class MetaspacePreTokenizer extends PreTokenizer {
	addPrefixSpace: boolean;
	replacement: string;
	strRep: string;
	prependScheme: 'first' | 'never' | 'always';
	/**
	 * @param {Object} config The configuration object for the MetaspacePreTokenizer.
	 * @param {boolean} config.addPrefixSpace Whether to add a prefix space to the first token.
	 * @param {string} config.replacement The character to replace spaces with.
	 * @param {string} [config.strRep=config.replacement] An optional string representation of the replacement character.
	 * @param {'first'|'never'|'always'} [config.prependScheme='always'] The metaspace prepending scheme.
	 */
	constructor(config: MetaspacePreTokenizerConfig) {
		super();

		this.addPrefixSpace = config.addPrefixSpace;
		this.replacement = config.replacement;
		this.strRep = config.strRep || this.replacement;
		this.prependScheme = config.prependScheme ?? 'always';
	}

	/**
	 * This method takes a string, replaces spaces with the replacement character,
	 * adds a prefix space if requested, and returns a new list of tokens.
	 * @param text The text to pre-tokenize.
	 * @param options The options for the pre-tokenization.
	 * @param options.sectionIndex The index of the section to pre-tokenize.
	 * @returns A new list of pre-tokenized tokens.
	 */
	preTokenizeText(
		text: string,
		{ sectionIndex: sectionIndex = undefined }: PreTokenizeOptions = {}
	) {
		let normalized = text.replaceAll(' ', this.strRep);

		if (
			// We add a prefix space if:
			//  (1) The addPrefixSpace option is enabled and the normalized token does not already start
			//      with the replacement character.
			this.addPrefixSpace &&
			!normalized.startsWith(this.replacement) &&
			// and (2) either:
			//  (a) prependScheme is 'always'
			//  (b) prependScheme is 'first' and this is the first section
			(this.prependScheme === 'always' || (this.prependScheme === 'first' && sectionIndex === 0))
		) {
			normalized = this.strRep + normalized;
		}
		return [normalized];
	}
}

/**
 * A pre-tokenizer that applies a sequence of pre-tokenizers to the input text.
 * @extends PreTokenizer
 */
class PreTokenizerSequence extends PreTokenizer {
	tokenizers: PreTokenizer[];
	/**
	 * Creates an instance of PreTokenizerSequence.
	 * @param {Object} config The configuration object for the pre-tokenizer sequence.
	 * @param {Object[]} config.pretokenizers An array of pre-tokenizer configurations.
	 */
	constructor(config: PreTokenizerSequenceConfig) {
		super();
		this.tokenizers = config.pretokenizers.map((x) => PreTokenizer.fromConfig(x));
	}

	/**
	 * Applies each pre-tokenizer in the sequence to the input text in turn.
	 * @param text The text to pre-tokenize.
	 * @param options Additional options for the pre-tokenization logic.
	 * @returns The pre-tokenized text.
	 */
	preTokenizeText(text: string, options: PreTokenizeOptions) {
		// Use reduce to apply each tokenizer to the text
		return this.tokenizers.reduce(
			(preTokenizedText, tokenizer) => {
				return tokenizer.preTokenize(preTokenizedText, options);
			},
			[text]
		);
	}
}

/**
 * Splits on word boundaries (using the following regular expression: `\w+|[^\w\s]+`).
 */
class WhitespacePreTokenizer extends PreTokenizer {
	/**
	 * Creates an instance of WhitespacePreTokenizer.
	 * @param config The configuration object for the pre-tokenizer.
	 */
	constructor() {
		super();
	}
	/**
	 * Pre-tokenizes the input text by splitting it on word boundaries.
	 * @param text The text to be pre-tokenized.
	 * @param options Additional options for the pre-tokenization logic.
	 * @returns An array of tokens produced by splitting the input text on whitespace.
	 */
	preTokenizeText(text: string, _options: unknown) {
		return text.match(/\w+|[^\w\s]+/g) || [];
	}
}

/**
 * Splits a string of text by whitespace characters into individual tokens.
 * @extends PreTokenizer
 */
class WhitespaceSplit extends PreTokenizer {
	/**
	 * Creates an instance of WhitespaceSplit.
	 * @param config The configuration object for the pre-tokenizer.
	 */
	constructor() {
		super();
	}
	/**
	 * Pre-tokenizes the input text by splitting it on whitespace characters.
	 * @param text The text to be pre-tokenized.
	 * @param options Additional options for the pre-tokenization logic.
	 * @returns An array of tokens produced by splitting the input text on whitespace.
	 */
	preTokenizeText(text: string, _options: unknown) {
		return whitespaceSplit(text);
	}
}

const SPECIAL_TOKEN_ATTRIBUTES = [
	'bos_token',
	'eos_token',
	'unk_token',
	'sep_token',
	'pad_token',
	'cls_token',
	'mask_token'
	// additional_special_tokens (TODO)
];

/**
 *
 * Helper function for padding values of an object, which are each arrays.
 * NOTE: No additional checks are made here for validity of arguments.
 * @param item The input object.
 * @param length The length to pad to.
 * @param valueFn Determine the value to fill the array, based on its key.
 * @param side Which side to pad the array.
 */
function padHelper<T>(
	item: Record<string, T[]>,
	length: number,
	valueFn: (key: string) => T,
	side: 'right' | 'left'
) {
	for (const key of Object.keys(item)) {
		const diff = length - item[key].length;
		const value = valueFn(key);

		const padData = new Array(diff).fill(value);
		item[key] =
			side === 'right' ? mergeArrays(item[key], padData) : mergeArrays(padData, item[key]);
	}
}

/**
 * Helper function for truncating values of an object, which are each arrays.
 * NOTE: No additional checks are made here for validity of arguments.
 * @param item The input object.
 * @param length The length to truncate to.
 */
function truncateHelper<T>(item: Record<string, T[]>, length: number) {
	// Setting .length to a lower value truncates the array in-place:
	// https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/Array/length
	for (const key of Object.keys(item)) {
		item[key].length = length;
	}
}

interface DecodeArgs {
	skipSpecialTokens?: boolean;
	cleanUpTokenizationSpaces?: boolean;
}

type BatchEncodingItem = number[] | number[][] | Tensor;

interface BatchEncoding {
	inputIds: BatchEncodingItem;
	attentionMask: BatchEncodingItem;
	tokenTypeIds?: BatchEncodingItem;
}

interface Message {
	role: string;
	content: string;
}

export class PreTrainedTokenizer extends Callable<
	[
		string | string[],
		{
			textPair?: string | null;
			addSpecialTokens?: boolean;
			padding?: boolean | 'max_length';
			truncation?: boolean | null;
			maxLength?: number | null;
			returnTensor?: boolean;
			returnTokenTypeIds?: boolean | null;
		}?
	],
	BatchEncoding
> {
	config: TokenizerConfig;
	normalizer!: ((text: string) => string) | Normalizer | null;
	preTokenizer!: ((text: string, options?: PreTokenizeOptions) => string[]) | PreTokenizer | null;
	model!: TokenizerModel;
	postProcessor!:
		| ((
				tokens: string[],
				tokensPair?: string[] | null,
				options?: PostProcessorOptions
		  ) => PostProcessorResult)
		| PostProcessor
		| null;
	decoder!: ((tokens: string[]) => string) | Decoder | null;
	specialTokens: string[];
	allSpecialIds: number[];
	addedTokens: AddedToken[];
	additionalSpecialTokens: string[];
	addedTokensSplitter: DictionarySplitter;
	addedTokensMap: Map<string, AddedToken>;
	maskToken?: string | null;
	maskTokenId?: number;
	padToken?: string | null;
	padTokenId?: number;
	sepToken?: string | null;
	sepTokenId?: number;
	unkToken?: string | null;
	unkTokenId?: number;
	bosToken?: string | null;
	bosTokenId?: number;
	eosToken?: string | null;
	eosTokenId?: number;
	modelMaxLength!: number;
	removeSpace!: boolean;
	cleanUpTokenizationSpaces!: boolean;
	paddingSide: 'left' | 'right' = 'right';
	addBoxToken?: boolean;
	addEosToken?: boolean;
	chatTemplate: null | Record<string, string> | Array<{ name: string; template: string }>;
	returnTokenTypeIds = false;
	private compiledTemplateCache: Map<string, Template>;
	constructor(tokenizerJSON: TokenizerJSON, tokenizerConfig: TokenizerConfig) {
		super();
		this.config = tokenizerConfig;
		this.normalizer = tokenizerJSON.normalizer
			? Normalizer.fromConfig(tokenizerJSON.normalizer)
			: null;
		this.preTokenizer = tokenizerJSON.preTokenizer
			? PreTokenizer.fromConfig(tokenizerJSON.preTokenizer)
			: null;
		this.model = TokenizerModel.fromConfig(tokenizerJSON.model, tokenizerConfig);
		this.postProcessor = tokenizerJSON.postProcessor
			? PostProcessor.fromConfig(tokenizerJSON.postProcessor)
			: null;
		this.decoder = tokenizerJSON.decoder ? Decoder.fromConfig(tokenizerJSON.decoder) : null;
		this.specialTokens = [];
		this.allSpecialIds = [];
		this.addedTokens = [];
		for (const addedToken of tokenizerJSON.addedTokens) {
			const token = new AddedToken(addedToken);
			this.addedTokens.push(token);
			this.model.tokensToIds.set(token.content, token.id);
			this.model.vocab[token.id] = token.content;
			if (token.special) {
				this.specialTokens.push(token.content);
				this.allSpecialIds.push(token.id);
			}
		}
		this.additionalSpecialTokens = tokenizerConfig.additionalSpecialTokens ?? [];
		this.specialTokens.push(...this.additionalSpecialTokens);
		this.specialTokens = [...new Set(this.specialTokens)];
		if (this.decoder) {
			(this.decoder as Decoder).addedTokens = this.addedTokens;
			(this.decoder as Decoder).endOfWordSuffix = this.model.endOfWordSuffix;
		}
		this.addedTokensSplitter = new DictionarySplitter(this.addedTokens.map((x) => x.content));
		this.addedTokensMap = new Map(this.addedTokens.map((x) => [x.content, x]));
		this.maskToken = this.getToken('mask_token');
		this.maskTokenId = this.model.tokensToIds.get(this.maskToken as string);
		this.padToken = this.getToken('pad_token', 'eos_token');
		this.padTokenId = this.model.tokensToIds.get(this.padToken as string);
		this.sepToken = this.getToken('sep_token');
		this.sepTokenId = this.model.tokensToIds.get(this.sepToken as string);
		this.unkToken = this.getToken('unk_token');
		this.unkTokenId = this.model.tokensToIds.get(this.unkToken as string);
		this.bosToken = this.getToken('bos_token');
		this.bosTokenId = this.model.tokensToIds.get(this.bosToken as string);
		this.eosToken = this.getToken('eos_token');
		this.eosTokenId = this.model.tokensToIds.get(this.eosToken as string);
		this.modelMaxLength = tokenizerConfig.modelMaxLength as number;
		this.removeSpace = tokenizerConfig.removeSpace as boolean;
		this.cleanUpTokenizationSpaces = (tokenizerConfig.cleanUpTokenizationSpaces ?? true) as boolean;
		if (tokenizerConfig.paddingSide) {
			this.paddingSide = tokenizerConfig.paddingSide as 'left' | 'right';
		}
		this.addBoxToken = tokenizerConfig.addBosToken as boolean;
		this.addEosToken = tokenizerConfig.addEosToken as boolean;
		this.chatTemplate = tokenizerConfig.chatTemplate ?? null;
		if (Array.isArray(this.chatTemplate)) {
			const chatTemplate: Record<string, string> = Object.create(null);
			for (const { name, template } of this.chatTemplate) {
				if (typeof name !== 'string' || typeof template !== 'string') {
					throw new Error(
						'Chat template must be a list of objects with "name" and "template" properties'
					);
				}
				chatTemplate[name] = template;
			}
			this.chatTemplate = chatTemplate;
		}
		this.compiledTemplateCache = new Map();
	}
	getToken(...keys: string[]): string | null {
		for (const key of keys) {
			const item = this.config[key];
			if (!item) continue;
			if (typeof item === 'object') {
				const maybe = item as { type?: string; content?: string };
				if (maybe.type === 'AddedToken' && typeof maybe.content === 'string') {
					return maybe.content;
				}
				throw Error(`Unknown token: ${String(item)}`);
			} else {
				return item as string;
			}
		}
		return null;
	}

	static async fromPretrained(tokenizerName: string): Promise<PreTrainedTokenizer> {
		const info = await loadTokenizer(tokenizerName);
		return new this(...info);
	}

	/**
	 * Encode/tokenize the given text(s).
	 * @param text The text to tokenize.
	 * @param options An optional object containing the following properties:
	 * @param options.textPair A second sequence to be encoded with the first.
	 * @param options.padding Whether to pad the input sequences.
	 * @param options.addSpecialTokens Whether or not to add the special tokens associated with the corresponding model.
	 * @param options.truncation Whether to truncate the input sequences.
	 * @param options.maxLength Maximum length of the returned list and optionally padding length.
	 * @param options.returnTensor Whether to return the results as Tensors or arrays.
	 * @param options.returnTokenTypeIds Whether to return the token type ids.
	 * @returns Object to be passed to the model.
	 */
	protected call(
		text: string | string[],
		{
			textPair = null,
			addSpecialTokens = true,
			padding = false,
			truncation = null,
			maxLength = null,
			returnTensor = true,
			returnTokenTypeIds = null
		}: {
			textPair?: string | null;
			addSpecialTokens?: boolean;
			padding?: boolean | 'max_length';
			truncation?: boolean | null;
			maxLength?: number | null;
			returnTensor?: boolean;
			returnTokenTypeIds?: boolean | null;
		} = {}
	): BatchEncoding {
		const isBatched = Array.isArray(text);

		let encodedTokens;

		if (isBatched) {
			if (text.length === 0) {
				throw Error('text array must be non-empty');
			}
			encodedTokens = text.map((x) =>
				this.encodePlus(x, {
					addSpecialTokens: addSpecialTokens,
					returnTokenTypeIds: returnTokenTypeIds
				})
			);
		} else {
			if (text === null || text === undefined) {
				throw Error('text may not be null or undefined');
			}

			if (Array.isArray(textPair)) {
				throw Error(
					'When specifying `textPair`, since `text` is a string, `textPair` must also be a string (i.e., not an array).'
				);
			}

			// For single input, we just wrap in an array, and then unwrap later.
			encodedTokens = [
				this.encodePlus(text, {
					addSpecialTokens: addSpecialTokens,
					returnTokenTypeIds: returnTokenTypeIds
				})
			];
		}
		// At this point, `encodedTokens` is batched, of shape [batchSize, tokens].
		// However, array may be jagged. So, we may need pad to maxLength.
		if (maxLength === null) {
			maxLength = this.modelMaxLength;
		} else if (truncation === null) {
			if (padding === true) {
				console.warn(
					'`maxLength` is ignored when `padding: true` and there is no truncation strategy. ' +
						"To pad to max length, use `padding: 'maxLength'`."
				);
				maxLength = this.modelMaxLength;
			} else if (padding === false) {
				console.warn(
					'Truncation was not explicitly activated but `maxLength` is provided a specific value, please use `truncation: true` to explicitly truncate examples to max length.'
				);
				truncation = true;
			}
		}

		// padding: 'maxLength' doesn't require any additional calculation
		// but padding: true has to calculate maxLength from the sequences
		if (padding === true) {
			maxLength = Math.min(
				max(encodedTokens.map((x) => x.inputIds.length))[0],
				maxLength ?? Infinity
			);
		}

		// Ensure it is less than model max length
		maxLength = Math.min(maxLength, this.modelMaxLength ?? Infinity);

		if (padding || truncation) {
			// Perform padding and/or truncation
			for (let i = 0; i < encodedTokens.length; ++i) {
				if (encodedTokens[i].inputIds.length === maxLength) {
					continue;
				} else if (encodedTokens[i].inputIds.length > maxLength) {
					// possibly truncate
					if (truncation) {
						truncateHelper(encodedTokens[i], maxLength);
					}
				} else {
					// t.length < maxLength
					// possibly pad
					if (padding) {
						padHelper(
							encodedTokens[i],
							maxLength,
							(key) => (key === 'inputIds' ? this.padTokenId : 0),
							this.paddingSide
						);
					}
				}
			}
		}

		const result: Record<string, unknown | Tensor[]> = {};

		if (returnTensor) {
			if (!(padding && truncation)) {
				// Not, guaranteed that all items have same length, so
				// we perform additional check

				if (
					encodedTokens.some((x) => {
						for (const key of Object.keys(x)) {
							if (
								(x as Record<string, unknown[]>)[key].length !==
								(encodedTokens[0] as Record<string, unknown[]>)[key]?.length
							) {
								return true;
							}
						}
						return false;
					})
				) {
					throw Error(
						'Unable to create tensor, you should probably activate truncation and/or padding ' +
							"with 'padding=true' and 'truncation=true' to have batched tensors with the same length."
					);
				}
			}

			// Now we actually convert to tensor
			// NOTE: In the same way as the python library, we return a batched tensor, regardless of
			// whether we have a single input or multiple inputs.
			const dims = [encodedTokens.length, encodedTokens[0].inputIds.length];

			for (const key of Object.keys(encodedTokens[0])) {
				result[key] = tensor(
					Int32Array.from(
						encodedTokens
							.flatMap(
								(x) =>
									(x as Record<string, unknown[]>)[key] as (bigint | boolean | number | string)[]
							)
							.map(Number)
					),
					{ shape: dims, dtype: int32 }
				);
			}
		} else {
			for (const key of Object.keys(encodedTokens[0])) {
				result[key] = encodedTokens.map((x) => (x as Record<string, unknown[]>)[key]);
			}

			// If not returning a tensor, we match the input type
			if (!isBatched) {
				// Input was not batched, so we unwrap
				for (const key of Object.keys(result)) {
					result[key] = (result[key] as unknown[])[0];
				}
			}
		}

		return result as unknown as BatchEncoding;
	}

	/**
	 * Encodes a single text using the preprocessor pipeline of the tokenizer.
	 *
	 * @param {string|null} text The text to encode.
	 * @returns {string[]|null} The encoded tokens.
	 */
	private encodeText(text: string | null): string[] | null {
		if (text === null) return null;

		// Actual function which does encoding, for a single text
		// First, we take care of special tokens. Needed to avoid issues arising from
		// normalization and/or pretokenization (which may not preserve special tokens)
		const sections = this.addedTokensSplitter.split(text);

		// Process left/right stripping of added tokens
		for (let i = 0; i < sections.length; ++i) {
			const addedToken = this.addedTokensMap.get(sections[i]);
			if (addedToken) {
				if (addedToken.lstrip && i > 0) {
					sections[i - 1] = sections[i - 1].trimEnd();
				}
				if (addedToken.rstrip && i < sections.length - 1) {
					sections[i + 1] = sections[i + 1].trimStart();
				}
			}
		}

		const tokens = sections.flatMap((x, sectionIndex) => {
			if (x.length === 0) return [];
			if (this.addedTokensMap.has(x)) return [x]; // Return added tokens unchanged

			if (this.removeSpace === true) {
				x = x.trim().split(/\s+/).join(' ');
			}

			if (this.normalizer !== null) {
				x = this.normalizer(x);
			}

			// If, after normalization, this section is empty (e.g., trimming whitespace),
			// we return an empty array
			if (x.length === 0) {
				return [];
			}

			const sectionTokens =
				this.preTokenizer !== null
					? this.preTokenizer(x, {
							sectionIndex: sectionIndex
						})
					: [x];

			const tokens = this.model(sectionTokens);

			return tokens;
		});

		return tokens;
	}

	/**
	 * Encodes a single text or a pair of texts using the model's tokenizer.
	 *
	 * @param text The text to encode.
	 * @param options An optional object containing the following properties:
	 * @param options.textPair The optional second text to encode.
	 * @param options.addSpecialTokens Whether or not to add the special tokens associated with the corresponding model.
	 * @param options.returnTokenTypeIds Whether to return tokenTypeIds.
	 * @returns An object containing the encoded text.
	 */
	private encodePlus(
		text: string,
		{
			textPair = null,
			addSpecialTokens = true,
			returnTokenTypeIds = null
		}: {
			textPair?: string | null;
			addSpecialTokens?: boolean;
			returnTokenTypeIds?: boolean | null;
		} = {}
	) {
		const { tokens, tokenTypeIds } = this.tokenizeHelper(text, {
			pair: textPair,
			addSpecialTokens
		});

		const inputIds = this.model.convertTokensToIds(tokens);

		const result = {
			inputIds: inputIds,
			attentionMask: new Array(inputIds.length).fill(1)
		};
		if ((returnTokenTypeIds ?? this.returnTokenTypeIds) && tokenTypeIds) {
			(result as { tokenTypeIds?: number[] }).tokenTypeIds = tokenTypeIds;
		}
		return result;
	}

	/**
	 * Internal helper function to tokenize a text, and optionally a pair of texts.
	 * @param text The text to tokenize.
	 * @param options An optional object containing the following properties:
	 * @param options.pair The optional second text to tokenize.
	 * @param options.addSpecialTokens Whether or not to add the special tokens associated with the corresponding model.
	 * @returns An object containing the tokens and optionally the token type IDs.
	 */
	private tokenizeHelper(
		text: string,
		{
			pair = null,
			addSpecialTokens = false
		}: { pair?: string | null; addSpecialTokens?: boolean } = {}
	) {
		const tokens = this.encodeText(text);
		const tokens2 = this.encodeText(pair);

		return this.postProcessor
			? this.postProcessor(tokens ?? [], tokens2 ?? null, { addSpecialTokens })
			: { tokens: mergeArrays(tokens ?? [], tokens2 ?? []) };
	}

	/**
	 * Converts a string into a sequence of tokens.
	 * @param text The sequence to be encoded.
	 * @param options An optional object containing the following properties:
	 * @param options.pair A second sequence to be encoded with the first.
	 * @param options.addSpecialTokens Whether or not to add the special tokens associated with the corresponding model.
	 * @returns The list of tokens.
	 */
	tokenize(text: string, { pair = null, addSpecialTokens = false } = {}) {
		return this.tokenizeHelper(text, { pair, addSpecialTokens }).tokens;
	}

	/**
	 * Encodes a single text or a pair of texts using the model's tokenizer.
	 *
	 * @param text The text to encode.
	 * @param options An optional object containing the following properties:
	 * @param options.addSpecialTokens Whether or not to add the special tokens associated with the corresponding model.
	 * @param options.returnTokenTypeIds Whether to return tokenTypeIds.
	 * @returns An array of token IDs representing the encoded text(s).
	 */
	encode(text: string, { addSpecialTokens = true, returnTokenTypeIds = null } = {}) {
		return this.encodePlus(text, {
			addSpecialTokens,
			returnTokenTypeIds
		}).inputIds;
	}

	/**
	 * Decode a batch of tokenized sequences.
	 * @param batch List of tokenized input sequences.
	 * @param decodeArgs (Optional) Object with decoding arguments.
	 * @returns List of decoded sequences.
	 */
	batchDecode(batch: number[][], decodeArgs: DecodeArgs = {}) {
		return batch.map((x) => this.decode(x, decodeArgs));
	}

	/**
	 * Decodes a sequence of token IDs back to a string.
	 *
	 * @param tokenIds List of token IDs to decode.
	 * @param decodeArgs (Optional) Object with decoding arguments.
	 *
	 * @returns The decoded string.
	 * @throws If `tokenIds` is not a non-empty array of integers.
	 */
	decode(tokenIds: number[], decodeArgs: DecodeArgs = {}) {
		if (
			!Array.isArray(tokenIds) ||
			tokenIds.length === 0 ||
			!(Number.isInteger(tokenIds[0]) || typeof tokenIds[0] === 'bigint')
		) {
			throw Error('tokenIds must be a non-empty array of integers.');
		}

		return this.decodeSingle(tokenIds, decodeArgs);
	}

	/**
	 * Decode a single list of token ids to a string.
	 * @param tokenIds List of token ids to decode
	 * @param decodeArgs Optional arguments for decoding
	 * @param [decodeArgs.skipSpecialTokens=false] Whether to skip special tokens during decoding
	 * @param [decodeArgs.cleanUpTokenizationSpaces=null] Whether to clean up tokenization spaces
	 * during decoding. If null, the value is set to `this.decoder.cleanup` if it exists, falling
	 * back to `this.cleanUpTokenizationSpaces` if it exists, falling back to `true`.
	 * @returns The decoded string
	 */
	decodeSingle(tokenIds: number[], { skipSpecialTokens = false }: DecodeArgs = {}) {
		let tokens = this.model.convertIdsToTokens(tokenIds);
		if (skipSpecialTokens) {
			tokens = tokens.filter((x) => !this.specialTokens.includes(x));
		}

		// If `this.decoder` is null, we just join tokens with a space:
		// https://github.com/huggingface/tokenizers/blob/8edec536a737cb04494b454805be16c020abb14f/tokenizers/src/tokenizer/mod.rs#L835
		let decoded = this.decoder ? this.decoder(tokens) : tokens.join(' ');

		// Slight hack, but prevents having to pass `skipSpecialTokens` to each call to `decode`, which
		// would lead to code duplication.
		if (this.decoder && 'endOfWordSuffix' in this.decoder && this.decoder.endOfWordSuffix) {
			decoded = decoded.replaceAll(this.decoder.endOfWordSuffix, ' ');
			if (skipSpecialTokens) {
				decoded = decoded.trim();
			}
		}

		return decoded;
	}

	/**
	 * Retrieve the chat template string used for tokenizing chat messages. This template is used
	 * internally by the `applyChatTemplate` method and can also be used externally to retrieve the
	 * model's chat template for better generation tracking.
	 *
	 * @param options An optional object containing the following properties:
	 * @param options.chatTemplate A Jinja template or the name of a template to use for this
	 * conversion. It is usually not necessary to pass anything to this argument, as the model's
	 * template will be used by default.
	 * @param options.tools A list of tools (callable functions) that will be accessible to the model.
	 * If the template does not support function calling, this argument will have no effect. Each
	 * tool should be passed as a JSON Schema, giving the name, description and argument types for
	 * the tool. See our
	 * [chat templating guide](https://huggingface.co/docs/transformers/main/en/chat_templating#automated-function-conversion-for-tool-use)
	 * for more information.
	 * @returns The chat template string.
	 */
	getChatTemplate({
		chatTemplate = null,
		tools = null
	}: { chatTemplate?: string | null; tools?: string[] | null } = {}): string {
		// First, handle the cases when the model has a dict of multiple templates
		if (this.chatTemplate && typeof this.chatTemplate === 'object') {
			const templateDict = this.chatTemplate;

			if (chatTemplate !== null && Object.hasOwn(templateDict, chatTemplate)) {
				// The user can pass the name of a template to the chat template argument instead of an
				// entire template
				chatTemplate = (templateDict as Record<string, string>)[chatTemplate];
			} else if (chatTemplate === null) {
				if (tools !== null && 'toolUse' in templateDict) {
					chatTemplate = templateDict['toolUse'];
				} else if ('default' in templateDict) {
					chatTemplate = templateDict['default'];
				} else {
					throw Error(
						`This model has multiple chat templates with no default specified! Please either pass` +
							` a chat template or the name of the template you wish to use to the 'chatTemplate'` +
							` argument. Available template names are ${Object.keys(templateDict).sort()}.`
					);
				}
			}
		} else if (chatTemplate === null) {
			// These are the cases when the model has a single template
			// priority: `chatTemplate` argument > `tokenizer.chatTemplate`
			if (this.chatTemplate) {
				chatTemplate = this.chatTemplate;
			} else {
				throw Error(
					'Cannot use applyChatTemplate() because tokenizer.chatTemplate is not set and no template ' +
						'argument was passed! For information about writing templates and setting the ' +
						'tokenizer.chatTemplate attribute, please see the documentation at ' +
						'https://huggingface.co/docs/transformers/main/en/chat_templating'
				);
			}
		}
		return chatTemplate;
	}

	/**
	 * Converts a list of message objects with `"role"` and `"content"` keys to a list of token
	 * ids. This method is intended for use with chat models, and will read the tokenizer's chat_template attribute to
	 * determine the format and control tokens to use when converting.
	 *
	 * See [here](https://huggingface.co/docs/transformers/chat_templating) for more information.
	 *
	 * @param conversation A list of message objects with `"role"` and `"content"` keys,
	 * representing the chat history so far.
	 * @param options An optional object containing the following properties:
	 * @param options.chatTemplate A Jinja template to use for this conversion. If
	 * this is not passed, the model's chat template will be used instead.
	 * @param options.tools A list of tools (callable functions) that will be accessible to the model.
	 * If the template does not support function calling, this argument will have no effect. Each
	 * tool should be passed as a JSON Schema, giving the name, description and argument types for
	 * the tool. See our
	 * [chat templating guide](https://huggingface.co/docs/transformers/main/en/chat_templating#automated-function-conversion-for-tool-use)
	 * for more information.
	 * @param options.documents A list of dicts representing documents that will be accessible to the model if it is performing RAG
	 * (retrieval-augmented generation). If the template does not support RAG, this argument will have no
	 * effect. We recommend that each document should be a dict containing "title" and "text" keys. Please
	 * see the RAG section of the [chat templating guide](https://huggingface.co/docs/transformers/main/en/chat_templating#arguments-for-RAG)
	 * for examples of passing documents with chat templates.
	 * @param options.addGenerationPrompt Whether to end the prompt with the token(s) that indicate
	 * the start of an assistant message. This is useful when you want to generate a response from the
	 * model. Note that this argument will be passed to the chat template, and so it must be supported
	 * in the template for this argument to have any effect.
	 * @param options.tokenize Whether to tokenize the output. If false, the output will be a string.
	 * @param options.padding Whether to pad sequences to the maximum length. Has no effect if tokenize is false.
	 * @param options.truncation Whether to truncate sequences to the maximum length. Has no effect if tokenize is false.
	 * @param options.maxLength Maximum length (in tokens) to use for padding or truncation. Has no effect if tokenize is false.
	 * If not specified, the tokenizer's `max_length` attribute will be used as a default.
	 * @param options.returnTensor Whether to return the output as a Tensor or an Array. Has no effect if tokenize is false.
	 * @param options.returnDict Whether to return a dictionary with named outputs. Has no effect if tokenize is false.
	 * @param options.tokenizerKwargs Additional options to pass to the tokenizer.
	 * @returns The tokenized output.
	 */
	applyChatTemplate(
		conversation: Message[],
		{
			tools = null,
			documents = null,
			chatTemplate = null,
			addGenerationPrompt = false,
			tokenize = true,
			padding = false,
			truncation = false,
			maxLength = null,
			returnTensor = true,
			returnDict = false,
			tokenizerKwargs = {},
			...kwargs
		}: {
			tools?: string[] | null;
			documents?: string[] | null;
			chatTemplate?: string | null;
			addGenerationPrompt?: boolean;
			tokenize?: boolean;
			padding?: boolean;
			truncation?: boolean;
			maxLength?: number | null;
			returnTensor?: boolean;
			returnDict?: boolean;
			tokenizerKwargs?: Record<string, unknown>;
		} = {}
	) {
		chatTemplate = this.getChatTemplate({ chatTemplate, tools });

		if (typeof chatTemplate !== 'string') {
			throw Error(`chat_template must be a string, but got ${typeof chatTemplate}`);
		}

		// Compilation function uses a cache to avoid recompiling the same template
		let compiledTemplate = this.compiledTemplateCache.get(chatTemplate);
		if (compiledTemplate === undefined) {
			compiledTemplate = new Template(chatTemplate);
			this.compiledTemplateCache.set(chatTemplate, compiledTemplate);
		}

		const specialTokensMap = Object.create(null);
		for (const key of SPECIAL_TOKEN_ATTRIBUTES) {
			const value = this.getToken(key);
			if (value) {
				specialTokensMap[key] = value;
			}
		}

		const rendered = compiledTemplate.render({
			messages: conversation,
			addGenerationPrompt,
			tools,
			documents,
			...specialTokensMap,
			...kwargs
		});

		if (tokenize) {
			const out = this.call(rendered, {
				addSpecialTokens: false,
				padding,
				truncation,
				maxLength,
				returnTensor,
				...tokenizerKwargs
			});
			return returnDict ? out : out.inputIds;
		}

		return rendered;
	}
}

export function max<T extends number[] | bigint[]>(arr: T) {
	if (arr.length === 0) throw Error('Array must not be empty');
	let max = arr[0];
	let indexOfMax = 0;
	for (let i = 1; i < arr.length; ++i) {
		if (arr[i] > max) {
			max = arr[i];
			indexOfMax = i;
		}
	}
	return [max, indexOfMax] as T extends bigint[] ? [bigint, number] : [number, number];
}

function mergeArrays<T extends unknown[]>(...arrs: T[]): T {
	return Array.prototype.concat.apply([], arrs) as T;
}

type TrieNode = {
	/**
	 * If this node marks the end of a word, this property will
	 * contain the complete word. Otherwise, it's undefined.
	 */
	end?: string;

	/**
	 * An index signature to represent child nodes. Each key is a
	 * character, and each value is the next TrieNode in the sequence.
	 * The value is a union to satisfy TypeScript's index signature rules.
	 */
	[key: string]: TrieNode | string | undefined;
};

/**
 * A data structure which uses a trie to split a string into tokens based on a dictionary.
 * It can also use a regular expression to preprocess the input text before splitting.
 *
 * NOTE: To ensure multi-byte characters are handled correctly, we operate at byte-level instead of character-level.
 */
class DictionarySplitter {
	trie: TrieNode;
	/**
	 * @param dictionary The dictionary of words to use for splitting.
	 */
	constructor(dictionary: string[]) {
		this.trie = this.buildTrie(dictionary);
	}

	/**
	 * Builds a trie from the given dictionary.
	 * @param dictionary The dictionary of words to build the trie from.
	 * @returns The root node of the trie.
	 */
	private buildTrie(dictionary: string[]) {
		const trie: TrieNode = Object.create(null);
		for (const word of dictionary) {
			let node = trie;
			for (let i = 0; i < word.length; ++i) {
				node = (node[word[i]] ??= Object.create(null)) as TrieNode;
			}
			node.end = word;
		}
		return trie;
	}

	/**
	 * Splits the input text into tokens based on the dictionary.
	 * @param {string} text The input text to split.
	 * @returns {string[]} An array of tokens.
	 */
	split(text: string): string[] {
		const result = [];
		const n = text.length;
		let start = 0;
		let i = 0;

		while (i < n) {
			let node = this.trie;
			let match = null;
			let j = i;

			while (j < n && (node = node[text[j]] as TrieNode)) {
				if (node.end) {
					// Always keep the last (i.e., longest) match.
					match = node.end;
				}
				++j;
			}

			if (match) {
				if (i > start) {
					result.push(text.slice(start, i));
				}
				result.push(match);
				i += match.length;
				start = i;
			} else {
				++i;
			}
		}
		if (start < n) {
			result.push(text.slice(start));
		}
		return result;
	}
}

/**
 * Efficient Heap-based Implementation of a Priority Queue.
 * It uses an array-based binary heap, where the root is at index `0`, and the
 * children of node `i` are located at indices `2i + 1` and `2i + 2`, respectively.
 *
 * Adapted from the following sources:
 * - https://stackoverflow.com/a/42919752/13989043 (original)
 * - https://github.com/belladoreai/llama-tokenizer-js (minor improvements)
 */
class PriorityQueue<T> {
	private heap: T[];
	private comparator: (a: T, b: T) => boolean;
	private maxSize: number;
	/**
	 * Create a new PriorityQueue.
	 * @param comparator Comparator function to determine priority. Defaults to a MaxHeap.
	 */
	constructor(comparator = (a: T, b: T) => a > b, maxSize = Infinity) {
		this.heap = [];
		this.comparator = comparator;
		this.maxSize = maxSize;
	}

	/**
	 * The size of the queue
	 */
	get size() {
		return this.heap.length;
	}

	/**
	 * Check if the queue is empty.
	 * @returns `true` if the queue is empty, `false` otherwise.
	 */
	isEmpty() {
		return this.size === 0;
	}

	/**
	 * Return the element with the highest priority in the queue.
	 * @returns The highest priority element in the queue.
	 */
	peek() {
		return this.heap[0];
	}

	/**
	 * Add one or more elements to the queue.
	 * @param values The values to push into the queue.
	 * @returns The new size of the queue.
	 */
	push(...values: T[]) {
		return this.extend(values);
	}

	/**
	 * Add multiple elements to the queue.
	 * @param values The values to push into the queue.
	 * @returns The new size of the queue.
	 */
	extend(values: T[]) {
		for (const value of values) {
			if (this.size < this.maxSize) {
				this.heap.push(value);
				this.siftUp();
			} else {
				// Get index of value with the lowest priority
				const smallest = this.smallest();

				// If the new value has higher priority than the smallest value in the heap
				// then replace the smallest value with the new value and update the heap
				if (this.comparator(value, this.heap[smallest])) {
					this.heap[smallest] = value;
					this.siftUpFrom(smallest);
				}
			}
		}
		return this.size;
	}

	/**
	 * Remove and return the element with the highest priority in the queue.
	 * @returns The element with the highest priority in the queue.
	 */
	pop() {
		const poppedValue = this.peek();
		const bottom = this.size - 1;
		if (bottom > 0) {
			this.swap(0, bottom);
		}
		this.heap.pop();
		this.siftDown();
		return poppedValue;
	}

	/**
	 * Replace the element with the highest priority in the queue with a new value.
	 * @param value The new value.
	 * @returns The replaced value.
	 */
	replace(value: T) {
		const replacedValue = this.peek();
		this.heap[0] = value;
		this.siftDown();
		return replacedValue;
	}

	/**
	 * Compute the index for the parent of the node at index `i`.
	 * @param i The index of the node to get the parent of.
	 * @returns The index of the parent node.
	 */
	private parent(i: number) {
		return ((i + 1) >>> 1) - 1;
	}

	/**
	 * Compute the index for the left child of the node at index `i`.
	 * @param i The index of the node to get the left child of.
	 * @returns The index of the left child.
	 *
	 */
	private left(i: number) {
		return (i << 1) + 1;
	}

	/**
	 * Compute the index for the right child of the node at index `i`.
	 * @param i The index of the node to get the right child of.
	 * @returns The index of the right child.
	 */
	private right(i: number) {
		return (i + 1) << 1;
	}

	/**
	 * Check if the element at index `i` is greater than the element at index `j`.
	 * @param i The index of the first element to compare.
	 * @param j The index of the second element to compare.
	 * @returns `true` if the element at index `i` is greater than the element at index `j`, `false` otherwise.
	 *
	 */
	private greater(i: number, j: number) {
		return this.comparator(this.heap[i], this.heap[j]);
	}

	/**
	 * Swap the elements at indices `i` and `j`.
	 * @param i The index of the first element to swap.
	 * @param j The index of the second element to swap.
	 *
	 */
	private swap(i: number, j: number) {
		const temp = this.heap[i];
		this.heap[i] = this.heap[j];
		this.heap[j] = temp;
	}

	/**
	 * Maintain the heap property by updating positions in the heap,
	 * starting at the last element and moving up the heap.
	 */
	private siftUp() {
		this.siftUpFrom(this.size - 1);
	}

	/**
	 * Helper function to sift up from a given node.
	 * @param node The index of the node to start sifting up from.
	 */
	private siftUpFrom(node: number) {
		while (node > 0 && this.greater(node, this.parent(node))) {
			this.swap(node, this.parent(node));
			node = this.parent(node);
		}
	}

	/**
	 * Maintain the heap property by updating positions in the heap,
	 * starting at the first element and moving down the heap.
	 */
	private siftDown() {
		let node = 0;
		while (
			(this.left(node) < this.size && this.greater(this.left(node), node)) ||
			(this.right(node) < this.size && this.greater(this.right(node), node))
		) {
			const maxChild =
				this.right(node) < this.size && this.greater(this.right(node), this.left(node))
					? this.right(node)
					: this.left(node);
			this.swap(node, maxChild);
			node = maxChild;
		}
	}

	/**
	 * Get the index of the smallest element in the heap. Since we use an array-based heap,
	 * the index can be computed without needing to traverse the heap.
	 */
	private smallest(): number {
		return 2 ** Math.floor(Math.log2(this.size)) - 1;
	}
}

/**
 * A simple Least Recently Used (LRU) cache implementation in JavaScript.
 * This cache stores key-value pairs and evicts the least recently used item
 * when the capacity is exceeded.
 */
class LRUCache<Key, Value> {
	capacity: number;
	cache: Map<Key, Value>;
	/**
	 * Creates an LRUCache instance.
	 * @param capacity The maximum number of items the cache can hold.
	 */
	constructor(capacity: number) {
		this.capacity = capacity;
		this.cache = new Map();
	}

	/**
	 * Retrieves the value associated with the given key and marks the key as recently used.
	 * @param key The key to retrieve.
	 * @returns The value associated with the key, or undefined if the key does not exist.
	 */
	get(key: Key) {
		if (!this.cache.has(key)) return undefined;
		const value = this.cache.get(key);
		this.cache.delete(key);
		this.cache.set(key, value as Value);
		return value;
	}

	/**
	 * Inserts or updates the key-value pair in the cache.
	 * If the key already exists, it is updated and marked as recently used.
	 * If the cache exceeds its capacity, the least recently used item is evicted.
	 * @param key The key to add or update.
	 * @param value The value to associate with the key.
	 */
	put(key: Key, value: Value) {
		if (this.cache.has(key)) {
			this.cache.delete(key);
		}
		this.cache.set(key, value);
		if (this.cache.size > this.capacity) {
			this.cache.delete(this.cache.keys().next().value as Key);
		}
	}

	/**
	 * Clears the cache.
	 */
	clear() {
		this.cache.clear();
	}
}

export function decodeSingle(
	value: number,
	tokenizer: PreTrainedTokenizer | ToyTokenizer | null
): string {
	if (tokenizer instanceof PreTrainedTokenizer) {
		return tokenizer
			.decodeSingle([value])
			.replaceAll('<|end_of_text|>', '▶️📄')
			.replaceAll('<|im_start|>', '▶️💬')
			.replaceAll('<|im_end|>', '⏹️💬');
	}
	return tokenizer?.ids?.[value] || `<${value}>`;
}
