import type { DataLoader } from '@piston-ml/piston-web';

import type {
	ToyAutoregressiveBatch,
	ToyBidirectionalBatch,
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

export type ToyBatchType<W> =
	| ToyBidirectionalBatch<W>
	| ToyEncoderDecoderBatch<W>
	| ToyAutoregressiveBatch<W>;

export type EncoderDecoderBatchType<W> = ToyEncoderDecoderBatch<W>;

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

export type ToyCollateFnType<W> = CollateFn<ToyCollateInput, ToyBatchType<W>>;

export type ToyDataloaderType<W> = DataLoader<ToySequence, ToyBatchType<W>>;

export type GeneratableModel = DecoderTransformer | EncoderTransformer | EncoderDecoderTransformer;
