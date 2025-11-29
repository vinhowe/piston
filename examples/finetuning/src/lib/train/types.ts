import type { DataLoader } from '@piston-ml/piston-web';

import type { NaturalLanguageAutoregressiveBatch } from './data/natural';
import type { GPT } from './model/gpt';

type CollateFn<B, T> = (batch: B) => T;

type NaturalCollateInput = number[][];

export type NaturalBatchType<W> = NaturalLanguageAutoregressiveBatch<W>;

export type AutoregressiveBatchType<W> = NaturalLanguageAutoregressiveBatch<W>;

export type NaturalAutoregressiveCollateFnType<W> = CollateFn<
	NaturalCollateInput,
	NaturalLanguageAutoregressiveBatch<W>
>;

export type AutoregressiveCollateFnType<W> = NaturalAutoregressiveCollateFnType<W>;

export type NaturalCollateFnType<W> = CollateFn<NaturalCollateInput, NaturalBatchType<W>>;

export type NaturalDataloaderType<W> = DataLoader<number[], NaturalBatchType<W>>;

export type AutoregressiveModelType = GPT;

export type GeneratableModel = AutoregressiveModelType;
