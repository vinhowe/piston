import type { DataLoader } from '@piston-ml/piston-web';

import type {
	NaturalLanguageAutoregressiveBatch,
	NaturalLanguageBidirectionalBatch,
	NaturalLanguageDataset
} from './data/natural';
import type {
	ToyAutoregressiveBatch,
	ToyBidirectionalBatch,
	ToyDatasetLike,
	ToyEncoderDecoderBatch,
	ToySequence
} from './data/toy/dataset';
import type {
	DecoderTransformer,
	EncoderDecoderTransformer,
	EncoderTransformer
} from './model/transformer';

type CollateFn<B, T> = (batch: B) => T;

type ToyCollateInput = ToySequence[];
type NaturalCollateInput = number[][];

export type NaturalBatchType<W> =
	| NaturalLanguageBidirectionalBatch<W>
	| NaturalLanguageAutoregressiveBatch<W>;

export type ToyBatchType<W> =
	| ToyBidirectionalBatch<W>
	| ToyEncoderDecoderBatch<W>
	| ToyAutoregressiveBatch<W>;

export type EncoderDecoderBatchType<W> = ToyEncoderDecoderBatch<W>;
export type BidirectionalBatchType<W> =
	| ToyBidirectionalBatch<W>
	| NaturalLanguageBidirectionalBatch<W>;
export type AutoregressiveBatchType<W> =
	| ToyAutoregressiveBatch<W>
	| NaturalLanguageAutoregressiveBatch<W>;

export type ToyAutoregressiveCollateFnType<W> = CollateFn<
	ToyCollateInput,
	ToyAutoregressiveBatch<W>
>;
export type ToyBidirectionalCollateFnType<W> = CollateFn<ToyCollateInput, ToyBidirectionalBatch<W>>;
export type ToyEncoderDecoderCollateFnType<W> = CollateFn<
	ToyCollateInput,
	ToyEncoderDecoderBatch<W>
>;

export type EncoderDecoderCollateFnType<W> = CollateFn<ToyCollateInput, ToyEncoderDecoderBatch<W>>;

export type NaturalCollateFnType<W> = CollateFn<NaturalCollateInput, NaturalBatchType<W>>;
export type ToyCollateFnType<W> = CollateFn<ToyCollateInput, ToyBatchType<W>>;
export type PistonCollateFnType<W> = NaturalCollateFnType<W> | ToyCollateFnType<W>;

export type NaturalDataloaderType<W> = DataLoader<number[], NaturalBatchType<W>>;
export type ToyDataloaderType<W> = DataLoader<ToySequence, ToyBatchType<W>>;
export type PistonDataloaderType<W> = ToyDataloaderType<W> | NaturalDataloaderType<W>;

export type GeneratableModel = DecoderTransformer | EncoderTransformer | EncoderDecoderTransformer;

export type PistonDatasetType = ToyDatasetLike | NaturalLanguageDataset;
