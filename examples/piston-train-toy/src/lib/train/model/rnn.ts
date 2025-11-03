/**
 * @fileoverview Decoder-only RNN models (LSTM/GRU) styled similarly to GPT
 */

import type {
	Config,
	LayerNormalizationConfig,
	MultiplicativeRNNAttentionConfig,
	RNNAttentionConfig
} from '$lib/workspace/config';
import type { CrossEntropyLoss, Tensor } from '@piston-ml/piston-web';

import * as piston from '@piston-ml/piston-web';
import { nn } from '@piston-ml/piston-web';

import { MLMHead } from './bidirectional';
import { buildRNNConfigCommon, type RNNModuleConfig } from './config';
import { createNorm, type NormModule } from './modules/norm';
import {
	buildLmHeadRNN,
	computeAutoregressiveCrossEntropyLoss,
	createCrossEntropyCriterion,
	createPossiblyOneHotRNNWordEmbedding,
	maskedFill,
	type RNNWordEmbedding
} from './utils';

export type RNNEncoderConfig = RNNModuleConfig & {
	typeVocabSize: number;
	nLayers: number;
	bidirectional: boolean;
	mlmHead: { present: boolean; shareEmbeddings: boolean };
};

export type RNNDecoderConfig = RNNModuleConfig & {
	nLayers: number;
};

export type RNNEncoderDecoderConfig = RNNModuleConfig & {
	nEncoderLayer: number;
	nDecoderLayer: number;
	bidirectional: boolean;
	encoderDecoderAttention: RNNAttentionConfig;
};

type RNNCellType = GRUCell | LSTMCell | RNNCell;

// Optional attention inputs for sequence decoding inside BaseRNN
type RNNForwardAttentionOptions = {
	attentionModule: AdditiveRNNAttention | MultiplicativeRNNAttention;
	encoderOutputs: Tensor;
	srcPaddingMask?: Tensor | null;
	outProjection: nn.Linear; // projects concat([hidden, context]) -> hidden
	inputFeedingProjection?: nn.Linear; // projects concat([x_t, prevContext]) -> x_t
	decoderStateProjection?: nn.Linear; // projects decoder state H -> baseHidden if needed
};

abstract class BaseRNN<CellType extends RNNCellType> extends nn.Module {
	protected readonly layer: nn.ModuleList<CellType[]>;
	protected readonly interLayerDropout?: nn.Dropout;
	protected readonly interLayerNorms?: nn.ModuleList<NormModule[]>;
	protected readonly nLayers: number;
	protected readonly bidirectional: boolean;
	protected readonly hiddenSize: number;
	protected readonly projectionSize?: number;
	protected readonly projections?: nn.ModuleList<
		nn.ModuleDict<{ forwardProj: nn.Linear; reverseProj?: nn.Linear }>[]
	>;

	constructor(
		inputSize: number,
		hiddenSize: number,
		layerNormalization: LayerNormalizationConfig,
		options: RNNOptions = {}
	) {
		super();
		const nLayers = options.nLayers ?? 1;
		const bidirectional = options.bidirectional ?? false;
		const dropout = options.dropout ?? 0;
		const projectionSize = options.projectionSize;

		this.nLayers = nLayers;
		this.bidirectional = bidirectional;
		this.hiddenSize = hiddenSize;
		this.projectionSize = projectionSize;

		const cells: CellType[] = [];
		for (let l = 0; l < nLayers; l++) {
			// Input size calculation: first layer uses inputSize, subsequent layers use previous layer
			// output size
			let layerInputSize: number;
			if (l === 0) {
				layerInputSize = inputSize;
			} else {
				// Previous layer output size depends on projection and bidirectionality
				const prevLayerOutputSize = projectionSize || hiddenSize;
				layerInputSize = bidirectional ? prevLayerOutputSize * 2 : prevLayerOutputSize;
			}
			cells.push(this.createCell(layerInputSize, hiddenSize, layerNormalization));
		}
		this.layer = new nn.ModuleList(cells);

		// Create projection layers if projectionSize is specified
		if (projectionSize) {
			const projLayers: nn.ModuleDict<{ forwardProj: nn.Linear; reverseProj?: nn.Linear }>[] = [];
			for (let l = 0; l < nLayers; l++) {
				const layerProjs: { forwardProj: nn.Linear; reverseProj?: nn.Linear } = {
					forwardProj: new nn.Linear(hiddenSize, projectionSize)
				};
				if (bidirectional) {
					layerProjs.reverseProj = new nn.Linear(hiddenSize, projectionSize);
				}
				projLayers.push(new nn.ModuleDict(layerProjs));
			}
			this.projections = new nn.ModuleList(projLayers);
		}

		if (layerNormalization.rnn.betweenLayers && nLayers > 1) {
			const normDim = (bidirectional ? 2 : 1) * (projectionSize ?? hiddenSize);
			const norms: nn.Module<unknown, unknown>[] = [];
			for (let l = 0; l < nLayers - 1; l++) {
				norms.push(
					createNorm(normDim, layerNormalization) as unknown as nn.Module<unknown, unknown>
				);
			}
			this.interLayerNorms = new nn.ModuleList(norms);
		}

		if (dropout > 0) {
			this.interLayerDropout = new nn.Dropout(dropout);
		}

		this.resetParameters();
	}

	resetParameters(): void {
		// This is consistent with what PyTorch does internally
		// Initialize only linear layers; leave norms at their defaults (gamma=1, beta=0)
		const stdv = 1.0 / Math.sqrt(this.hiddenSize);
		this.apply((module: nn.Module) => {
			if (module instanceof nn.Linear) {
				piston.initUniform_(module.weight, { low: -stdv, high: stdv });
				if (module.bias) {
					piston.initUniform_(module.bias, { low: -stdv, high: stdv });
				}
			}
		});
	}

	protected abstract createCell(
		inputSize: number,
		hiddenSize: number,
		layerNormalization: LayerNormalizationConfig
	): CellType;

	private stepThroughCell(
		cell: CellType,
		x_t: Tensor,
		hPrev: Tensor,
		cPrev: Tensor | null
	): [Tensor, Tensor | null] {
		if (cell instanceof LSTMCell) {
			const [hNew, cNew] = cell.forward(x_t, hPrev, cPrev!);
			return [hNew, cNew];
		} else {
			const hNew = (cell as GRUCell | RNNCell).forward(x_t, hPrev);
			return [hNew, null];
		}
	}

	private applyProjection(
		projection: nn.Linear | undefined,
		outputs: Tensor,
		batchSize: number,
		seqLen: number
	): Tensor {
		if (!projection) {
			return outputs;
		}
		return projection
			.forward(outputs.view([-1, this.hiddenSize]))
			.view([batchSize, seqLen, this.projectionSize!]);
	}

	forward(
		x: Tensor,
		initial?: Parameters<CellType['forward']>[1],
		attention?: RNNForwardAttentionOptions
	): [Tensor, Tensor, Tensor | null] {
		if (attention && this.bidirectional) {
			throw new Error('Attention decoding with bidirectional RNN is not supported');
		}

		let layerInput = x;

		// Track final states for each layer
		const finalHiddenStates: Tensor[] = [];
		const finalCellStates: (Tensor | null)[] = [];

		for (let l = 0; l < this.nLayers; l++) {
			const [layerOutput, layerFinalH, layerFinalC] = this.runSingleLayer(
				layerInput,
				l,
				l === this.nLayers - 1 ? initial : undefined,
				l === this.nLayers - 1 ? attention : undefined
			);

			finalHiddenStates.push(layerFinalH);
			finalCellStates.push(layerFinalC);

			// Apply between-layer normalization and dropout (excluding the last layer)
			if (l < this.nLayers - 1) {
				let nextInput: Tensor = layerOutput;
				if (this.interLayerNorms) {
					nextInput = this.interLayerNorms[l]!.forward(nextInput) as Tensor;
				}
				if (this.interLayerDropout) {
					nextInput = this.interLayerDropout.forward(nextInput) as Tensor;
				}
				layerInput = nextInput;
			} else {
				layerInput = layerOutput;
			}
		}

		// Return the output of the last layer and concatenated final states
		const finalH = piston.cat(finalHiddenStates, { dim: 0 });
		let finalC: Tensor | null = null;
		if (finalCellStates.some((c) => c !== null)) {
			const validCellStates = finalCellStates.filter((c) => c !== null) as Tensor[];
			if (validCellStates.length > 0) {
				finalC = piston.cat(validCellStates, { dim: 0 });
			}
		}

		return [layerInput, finalH, finalC];
	}

	private runSingleLayer(
		yIn: Tensor,
		layerIdx: number,
		initial?: Tensor | [Tensor, Tensor],
		attention?: RNNForwardAttentionOptions
	): [Tensor, Tensor, Tensor | null] {
		const [batchSize, seqLen, _hiddenSize] = yIn.size();
		const cell = this.layer[layerIdx]!;

		// Get the projection module dict for this layer
		const layerProjections =
			this.projectionSize && this.projections ? this.projections[layerIdx] : null;

		const initializeStates = (
			initialForLayer?: Tensor | [Tensor, Tensor]
		): {
			hState: Tensor;
			cState: Tensor | null;
		} => {
			if (initialForLayer) {
				if (Array.isArray(initialForLayer)) {
					return { hState: initialForLayer[0], cState: initialForLayer[1] };
				} else {
					return { hState: initialForLayer, cState: null };
				}
			} else {
				// Create initial state directly with correct hidden size (avoid slicing yIn)
				const hState = piston.zeros([batchSize, this.hiddenSize], { device: yIn.device });
				const cState = cell instanceof LSTMCell ? piston.zerosLike(hState) : null;
				return { hState, cState };
			}
		};

		const runDirection = (
			seq: Tensor,
			allowAttention: boolean,
			initialForLayer?: Tensor | [Tensor, Tensor]
		): [Tensor, Tensor, Tensor | null] => {
			const { hState, cState } = initializeStates(initialForLayer);
			let currentH = hState;
			let currentC = cState;
			let prevContext: Tensor | null = null;
			const outputs: Tensor[] = [];

			for (let t = 0; t < seqLen; t++) {
				let x_t = seq
					.slice([
						[0, batchSize],
						[t, t + 1],
						[0, seq.size(2)]
					])
					.squeeze({ dim: 1 });

				if (allowAttention && attention && attention.inputFeedingProjection && prevContext) {
					const cat = piston.cat([x_t, prevContext], { dim: 1 });
					x_t = attention.inputFeedingProjection.forward(cat);
				}

				const [newH, newC] = this.stepThroughCell(cell, x_t, currentH, currentC);
				currentH = newH;
				currentC = newC;

				let stepOut: Tensor;
				if (allowAttention && attention) {
					const query = attention.decoderStateProjection
						? attention.decoderStateProjection.forward(currentH)
						: currentH;
					const [context] = attention.attentionModule.forward(
						query,
						attention.encoderOutputs,
						attention.srcPaddingMask ?? null
					);
					const combined = piston.cat([query, context], { dim: 1 });
					stepOut = attention.outProjection.forward(combined);
					prevContext = context;
				} else {
					stepOut = currentH;
				}

				outputs.push(stepOut.unsqueeze(1));
			}

			const y = piston.cat(outputs, { dim: 1 });
			return [y, currentH, currentC];
		};

		// Forward direction
		const [yF, lastHF, lastCF] = runDirection(yIn, !!attention, initial);

		const projectedOutputsForward = this.applyProjection(
			layerProjections?.dict.forwardProj,
			yF,
			batchSize,
			seqLen
		);
		const projectedLastHForward = layerProjections?.dict.forwardProj?.forward(lastHF) ?? lastHF;

		if (!this.bidirectional) {
			return [projectedOutputsForward, projectedLastHForward, lastCF];
		}

		// Backward direction (reverse sequence)
		const [yB, lastHB, lastCB] = runDirection(yIn.flip(1), false);
		// Flip back to align with forward
		const yBFlipped = yB.flip(1);

		// Apply projections if specified
		const projectedOutputsReverse = this.applyProjection(
			layerProjections?.dict.reverseProj,
			yBFlipped,
			batchSize,
			seqLen
		);

		const projectedLastHB = layerProjections?.dict.reverseProj?.forward(lastHB) ?? lastHB;

		// Concatenate forward and backward outputs
		const y = piston.cat([projectedOutputsForward, projectedOutputsReverse], { dim: 2 });
		const lastH = piston.cat([projectedLastHForward, projectedLastHB], { dim: 1 });

		let lastC: Tensor | null = null;
		if (lastCF && lastCB) {
			lastC = piston.cat([lastCF, lastCB], { dim: 1 });
		}

		return [y, lastH, lastC];
	}
}

export class GRUCell extends nn.Module {
	readonly WGates: nn.Linear;
	readonly WCandidate: nn.Linear;
	readonly normGates?: NormModule;
	readonly normCandidate?: NormModule;

	constructor(inputSize: number, hiddenSize: number, layerNormalization: LayerNormalizationConfig) {
		super();

		this.WGates = new nn.Linear(inputSize + hiddenSize, 2 * hiddenSize);
		this.WCandidate = new nn.Linear(inputSize + hiddenSize, hiddenSize);

		if (layerNormalization.rnn.withinCell) {
			this.normGates = createNorm(2 * hiddenSize, layerNormalization);
			this.normCandidate = createNorm(hiddenSize, layerNormalization);
		}
	}

	// Unified step API used by sequence runners
	forward(x: Tensor, hPrev: Tensor): Tensor {
		const combined = piston.cat([x, hPrev], { dim: 1 });
		let gates = this.WGates.forward(combined);
		if (this.normGates) {
			gates = this.normGates.forward(gates);
		}
		let [r, z] = piston.chunk(gates, 2, { dim: 1 });
		r = piston.sigmoid(r);
		z = piston.sigmoid(z);
		const combinedCandidate = piston.cat([x, r.mul(hPrev)], { dim: 1 });
		let candidate = this.WCandidate.forward(combinedCandidate);
		if (this.normCandidate) {
			candidate = this.normCandidate.forward(candidate);
		}
		const hTilde = piston.tanh(candidate);
		const hNext = hPrev.mul(z.neg().add(1)).add(hTilde.mul(z));
		return hNext;
	}
}

export class LSTMCell extends nn.Module {
	readonly W: nn.Linear;
	readonly normGates?: NormModule;
	readonly normCell?: NormModule;

	constructor(inputSize: number, hiddenSize: number, layerNormalization: LayerNormalizationConfig) {
		super();
		this.W = new nn.Linear(inputSize + hiddenSize, 4 * hiddenSize);
		if (layerNormalization.rnn.withinCell) {
			this.normGates = createNorm(4 * hiddenSize, layerNormalization);
			this.normCell = createNorm(hiddenSize, layerNormalization);
		}
	}

	// Unified step API used by sequence runners
	forward(x: Tensor, hPrev: Tensor, cPrev: Tensor): [Tensor, Tensor] {
		const combined = piston.cat([x, hPrev], { dim: 1 });
		let gateOutputs = this.W.forward(combined);
		if (this.normGates) {
			gateOutputs = this.normGates.forward(gateOutputs) as Tensor;
		}
		let [i, f, g, o] = piston.chunk(gateOutputs, 4, { dim: 1 });
		i = piston.sigmoid(i);
		f = piston.sigmoid(f);
		g = piston.tanh(g);
		o = piston.sigmoid(o);
		let cNext = f.mul(cPrev).add(i.mul(g));
		if (this.normCell) {
			cNext = this.normCell.forward(cNext) as Tensor;
		}
		const hNext = o.mul(piston.tanh(cNext));
		return [hNext, cNext];
	}
}

export class RNNCell extends nn.Module {
	readonly WIh: nn.Linear;
	readonly WHh: nn.Linear;
	readonly normPreact?: NormModule;
	constructor(inputSize: number, hiddenSize: number, layerNormalization: LayerNormalizationConfig) {
		super();
		this.WIh = new nn.Linear(inputSize, hiddenSize);
		this.WHh = new nn.Linear(hiddenSize, hiddenSize);
		if (layerNormalization.rnn.withinCell) {
			this.normPreact = createNorm(hiddenSize, layerNormalization);
		}
	}

	// Unified step API used by sequence runners
	forward(x: Tensor, hPrev: Tensor): Tensor {
		let preact = this.WIh.forward(x).add(this.WHh.forward(hPrev));
		if (this.normPreact) {
			preact = this.normPreact.forward(preact) as Tensor;
		}
		return preact.tanh();
	}
}

export interface RNNOptions {
	nLayers?: number;
	bidirectional?: boolean;
	dropout?: number;
	projectionSize?: number;
}

class GRU extends BaseRNN<GRUCell> {
	protected createCell(
		inputSize: number,
		hiddenSize: number,
		layerNormalization: LayerNormalizationConfig
	): GRUCell {
		return new GRUCell(inputSize, hiddenSize, layerNormalization);
	}
}

class LSTM extends BaseRNN<LSTMCell> {
	protected createCell(
		inputSize: number,
		hiddenSize: number,
		layerNormalization: LayerNormalizationConfig
	): LSTMCell {
		return new LSTMCell(inputSize, hiddenSize, layerNormalization);
	}
}

class RNN extends BaseRNN<RNNCell> {
	protected createCell(
		inputSize: number,
		hiddenSize: number,
		layerNormalization: LayerNormalizationConfig
	): RNNCell {
		return new RNNCell(inputSize, hiddenSize, layerNormalization);
	}
}

export function createRNN(
	cellType: 'gru' | 'lstm' | 'rnn',
	inputSize: number,
	hiddenSize: number,
	layerNormalization: LayerNormalizationConfig,
	options?: RNNOptions
): GRU | LSTM | RNN {
	switch (cellType) {
		case 'gru':
			return new GRU(inputSize, hiddenSize, layerNormalization, options);
		case 'lstm':
			return new LSTM(inputSize, hiddenSize, layerNormalization, options);
		case 'rnn':
			return new RNN(inputSize, hiddenSize, layerNormalization, options);
		default:
			throw new Error(`Invalid RNN cell type: ${cellType}`);
	}
}

class AdditiveRNNAttention extends nn.Module {
	private readonly Wh: nn.Linear;
	private readonly Ws: nn.Linear;
	private readonly v: nn.Linear; // projects attn dim -> 1

	constructor(hiddenSize: number, attnDim: number) {
		super();
		this.Wh = new nn.Linear(hiddenSize, attnDim);
		this.Ws = new nn.Linear(hiddenSize, attnDim);
		this.v = new nn.Linear(attnDim, 1);
	}

	forward(
		decoderState: Tensor,
		encoderOutputs: Tensor,
		srcPaddingMask: Tensor | null = null
	): [Tensor, Tensor] {
		const [B, S, _H] = encoderOutputs.size();
		const projEnc = this.Wh.forward(encoderOutputs); // [B,S,A]
		const projDec = this.Ws.forward(decoderState)
			.unsqueeze(1)
			.broadcastTo([B, S, projEnc.size(2)]); // [B,S,A]
		const e = projEnc.add(projDec).tanh(); // [B,S,A]
		let scores = this.v.forward(e).squeeze({ dim: -1 }); // [B,S]
		if (srcPaddingMask) {
			const mask = srcPaddingMask;
			scores = maskedFill(scores, mask, -1e9);
		}
		const alpha = scores.softmax(1); // [B,S]
		const context = encoderOutputs.mul(alpha.unsqueeze(-1)).sum({ dim: 1 }); // [B,H]
		return [context, alpha];
	}
}

class MultiplicativeRNNAttention extends nn.Module {
	private readonly Wa: nn.Linear; // general form s^T Wa h
	private readonly config: MultiplicativeRNNAttentionConfig;
	constructor(hiddenSize: number, config: MultiplicativeRNNAttentionConfig) {
		super();
		this.Wa = new nn.Linear(hiddenSize, hiddenSize, false);
		this.config = config;
	}

	forward(
		decoderState: Tensor,
		encoderOutputs: Tensor,
		srcPaddingMask: Tensor | null = null
	): [Tensor, Tensor] {
		const [_B, _S, H] = encoderOutputs.size();
		const encProj = this.Wa.forward(encoderOutputs); // [B,S,H]
		let scores = encProj.mul(decoderState.unsqueeze(1)).sum({ dim: -1 }); // [B,S]
		if (this.config.scaleByInverseSqrtHiddenSize) {
			scores = scores.div(Math.sqrt(H));
		}
		if (srcPaddingMask) {
			const mask = srcPaddingMask;
			scores = maskedFill(scores, mask, -1e9);
		}
		const alpha = scores.softmax(1); // [B,S]
		const context = encoderOutputs.mul(alpha.unsqueeze(-1)).sum({ dim: 1 }); // [B,H]
		return [context, alpha];
	}
}

export class RNNDecoder extends nn.Module {
	public config: RNNDecoderConfig;
	private readonly wordEmbeddings: RNNWordEmbedding;
	private readonly embeddingDropout?: nn.Dropout;
	private readonly decoder: GRU | LSTM | RNN;
	readonly lmHead: nn.Module<[Tensor], Tensor>;
	private readonly criterion: CrossEntropyLoss;

	constructor(config: Config, vocabSize: number) {
		super();

		this.config = { ...buildRNNConfigCommon(config, vocabSize), nLayers: config.model.layers };

		this.embeddingDropout = this.config.dropout.present
			? new nn.Dropout(this.config.dropout.embedding)
			: undefined;

		// Embedding selection: learned vs one-hot
		this.wordEmbeddings = createPossiblyOneHotRNNWordEmbedding(
			this.config.embedding,
			this.config.vocabSize,
			this.config.embeddingSize
		);

		// Build the RNN stack once so parameters are registered and trainable
		this.decoder = createRNN(
			this.config.cellType,
			this.config.embeddingSize,
			this.config.hiddenSize,
			this.config.layerNormalization,
			{
				nLayers: this.config.nLayers,
				// Decoder-only, so no bidirectional
				bidirectional: false,
				dropout: this.config.dropout.rnn.interLayer,
				projectionSize: this.config.projectionSize
			}
		);

		this.lmHead = buildLmHeadRNN(this.config, this.wordEmbeddings);
		this.criterion = createCrossEntropyCriterion(config);
	}

	/**
	 * @param input - Input tensor of token IDs [batch_size, seq_len]
	 * @param targets - Target tensor of token IDs [batch_size, seq_len]
	 * @returns [logits, loss]
	 */
	forward(
		input: Tensor,
		{ targets }: { targets: Tensor | null } = { targets: null }
	): [Tensor, Tensor | null] {
		const [_batchSize, seqLen] = input.size();
		if (!seqLen) {
			throw new Error(
				'Input tensor has no sequence length (did you forget to pass input as batches?)'
			);
		}

		// Embedding
		let embeddings = this.wordEmbeddings.forward(input);

		if (this.embeddingDropout) {
			embeddings = this.embeddingDropout.forward(embeddings) as Tensor;
		}

		// Process full sequence via pre-built core RNN
		const [hiddenStatesStacked] = this.decoder.forward(embeddings);

		// Project to vocabulary
		const logits = this.lmHead.forward(hiddenStatesStacked);

		const loss = computeAutoregressiveCrossEntropyLoss(logits, targets, this.criterion);

		return [logits, loss ?? null];
	}
}

export class RNNEncoder extends nn.Module {
	public config: RNNEncoderConfig;
	private readonly wordEmbedding: RNNWordEmbedding;
	private readonly tokenTypeEmbeddings?: nn.Embedding;
	private readonly embeddingDropout?: nn.Dropout;
	private readonly encoder: GRU | LSTM | RNN;
	private readonly criterion: CrossEntropyLoss;
	private readonly mlmHead?: MLMHead;
	private readonly mlmPreProj?: nn.Linear;

	constructor(config: Config, vocabSize: number) {
		super();

		this.config = {
			...buildRNNConfigCommon(config, vocabSize),
			nLayers: config.model.layers,
			bidirectional: config.model.rnn.encoder.bidirectional,
			typeVocabSize: 2,
			mlmHead: {
				present: true,
				shareEmbeddings: config.model.tieEmbeddingsAndLmHead
			}
		};

		this.embeddingDropout = this.config.dropout.present
			? new nn.Dropout(this.config.dropout.embedding)
			: undefined;
		this.wordEmbedding = createPossiblyOneHotRNNWordEmbedding(
			this.config.embedding,
			this.config.vocabSize,
			this.config.embeddingSize
		);
		this.tokenTypeEmbeddings = new nn.Embedding(
			this.config.typeVocabSize,
			this.config.embeddingSize
		);

		// Build the encoder RNN once so params are registered
		this.encoder = createRNN(
			this.config.cellType,
			this.config.embeddingSize,
			this.config.hiddenSize,
			this.config.layerNormalization,
			{
				nLayers: this.config.nLayers,
				bidirectional: this.config.bidirectional,
				dropout: this.config.dropout.rnn.interLayer,
				projectionSize: this.config.hiddenStateProjection.present
					? this.config.hiddenStateProjection.size
					: undefined
			}
		);

		if (this.config.mlmHead.present) {
			const finalHiddenSize = this.config.bidirectional
				? this.config.baseHiddenSize * 2
				: this.config.baseHiddenSize;
			const canTie =
				this.config.embedding.type === 'learned' && this.config.mlmHead.shareEmbeddings;
			// If tying and dims differ, add pre-projection to embedding size
			if (canTie && finalHiddenSize !== this.config.embeddingSize) {
				this.mlmPreProj = new nn.Linear(finalHiddenSize, this.config.embeddingSize);
			}
			this.mlmHead = new MLMHead(
				canTie ? this.config.embeddingSize : finalHiddenSize,
				this.config.vocabSize,
				this.config.layerNormalization,
				canTie ? (this.wordEmbedding as nn.Embedding) : undefined
			);
		}

		this.criterion = createCrossEntropyCriterion(config);
	}

	forward(
		inputIds: Tensor,
		{ tokenTypeIds, targets }: { tokenTypeIds?: Tensor | null; targets?: Tensor | null } = {
			tokenTypeIds: null,
			targets: null
		}
	): [Tensor, Tensor | null, Tensor | null] {
		const [batchSize, seqLen] = inputIds.size();
		if (!seqLen) {
			throw new Error(
				'Input tensor has no sequence length (did you forget to pass input as batches?)'
			);
		}

		// Get token embeddings
		let wordEmbeddings = this.wordEmbedding.forward(inputIds);

		// Add segment/token type embeddings
		if (this.tokenTypeEmbeddings) {
			if (tokenTypeIds == null) {
				tokenTypeIds = piston.zeros([batchSize, seqLen], {
					device: inputIds.device,
					dtype: piston.int32
				});
			}
			const typeEmbeddings = this.tokenTypeEmbeddings.forward(tokenTypeIds!);
			wordEmbeddings = wordEmbeddings.add(typeEmbeddings);
		}

		if (this.embeddingDropout) {
			wordEmbeddings = this.embeddingDropout.forward(wordEmbeddings) as Tensor;
		}

		let x = wordEmbeddings;
		const [xSeq] = this.encoder.forward(x);
		x = xSeq;

		// If present, reconcile dims for tied MLM head
		if (this.mlmPreProj) {
			x = this.mlmPreProj.forward(x) as Tensor;
		}

		// MLM head
		let mlmLogits: Tensor | null = null;
		if (this.mlmHead) {
			mlmLogits = this.mlmHead.forward(x);
		}

		// Calculate MLM loss if targets are provided
		let loss: Tensor | null = null;
		if (targets && mlmLogits) {
			// Only compute loss on masked positions (where targets != -100)
			const flatLogits = mlmLogits.view([-1, mlmLogits.size(-1)]);
			const flatTargets = targets.view(-1);
			loss = this.criterion.forward(flatLogits, flatTargets);
		}

		return [x, mlmLogits, loss];
	}
}

export class RNNEncoderDecoder extends nn.Module {
	public config: RNNEncoderDecoderConfig;

	private readonly criterion: CrossEntropyLoss;
	readonly lmHead: nn.Module<[Tensor], Tensor>;
	private readonly encoderEmbedding: RNNWordEmbedding;
	private readonly decoderEmbedding: RNNWordEmbedding;
	private readonly encoder: GRU | LSTM | RNN;
	private readonly decoder: GRU | LSTM | RNN;
	private readonly attention?: AdditiveRNNAttention | MultiplicativeRNNAttention;
	private readonly outProj: nn.Linear;
	private readonly decoderInputProj?: nn.Linear;
	private readonly decoderStateProj?: nn.Linear;
	private readonly encCombineProjections?: nn.ModuleList<nn.Linear[]>;

	constructor(config: Config, vocabSize: number) {
		super();

		this.config = {
			...buildRNNConfigCommon(config, vocabSize),
			nEncoderLayer: config.model.encoderDecoder.encoderLayers,
			nDecoderLayer: config.model.encoderDecoder.decoderLayers,
			bidirectional: config.model.rnn.encoder.bidirectional,
			encoderDecoderAttention: config.model.rnn.encoderDecoderAttention
		};

		// Effective hidden width used throughout encoder/decoder stacks
		const baseHidden = this.config.baseHiddenSize;

		this.encoderEmbedding = createPossiblyOneHotRNNWordEmbedding(
			this.config.embedding,
			this.config.vocabSize,
			this.config.embeddingSize
		);
		this.decoderEmbedding = createPossiblyOneHotRNNWordEmbedding(
			this.config.embedding,
			this.config.vocabSize,
			this.config.embeddingSize
		);

		// Build encoder/decoder stacks once so params are registered
		this.encoder = createRNN(
			this.config.cellType,
			this.config.embeddingSize,
			this.config.hiddenSize,
			this.config.layerNormalization,
			{
				nLayers: this.config.nEncoderLayer,
				bidirectional: this.config.bidirectional,
				dropout: this.config.dropout.rnn.interLayer,
				projectionSize: this.config.projectionSize
			}
		);

		this.decoder = createRNN(
			this.config.cellType,
			this.config.embeddingSize,
			this.config.hiddenSize,
			this.config.layerNormalization,
			{
				nLayers: this.config.nDecoderLayer,
				bidirectional: false,
				dropout: this.config.dropout.rnn.interLayer,
				projectionSize: this.config.projectionSize
			}
		);

		// Encoder bidirectionality support: per-layer 2H->H projections
		if (this.config.bidirectional) {
			this.encCombineProjections = new nn.ModuleList(
				Array.from(
					{ length: this.config.nEncoderLayer },
					() => new nn.Linear(baseHidden * 2, baseHidden) as nn.Module
				)
			);
		}

		// Attention selection
		if (config.model.rnn.encoderDecoderAttention?.present) {
			const attnType = config.model.rnn.encoderDecoderAttention.type;
			if (attnType === 'additive') {
				this.attention = new AdditiveRNNAttention(baseHidden, baseHidden);
			} else {
				this.attention = new MultiplicativeRNNAttention(
					baseHidden,
					this.config.encoderDecoderAttention.multiplicative
				);
			}
			if (config.model.rnn.encoderDecoderAttention.inputFeedingProjection) {
				const topIn =
					this.config.nDecoderLayer === 1 ? this.config.embeddingSize : this.config.baseHiddenSize;
				this.decoderInputProj = new nn.Linear(topIn + baseHidden, topIn);
			}
		}

		// Attention out projection maps back to the cell hidden size; projection (if any)
		// will be applied afterwards by BaseRNN
		this.outProj = new nn.Linear(baseHidden * 2, this.config.hiddenSize);

		// If hidden state differs from baseHidden (due to projection), project decoder
		// state to baseHidden for attention computations
		if (this.config.hiddenSize !== baseHidden) {
			this.decoderStateProj = new nn.Linear(this.config.hiddenSize, baseHidden);
		}
		this.lmHead = buildLmHeadRNN(this.config, this.decoderEmbedding);

		this.criterion = createCrossEntropyCriterion(config);
	}

	encode(inputIdsSrc: Tensor): Tensor {
		const [_batchSize, seqLen] = inputIdsSrc.size();
		if (!seqLen) {
			throw new Error('Source input tensor has no sequence length');
		}
		const x = this.encoderEmbedding.forward(inputIdsSrc); // [B,S,H]
		const [encSeq] = this.encoder.forward(x); // [B,S, H or 2H]
		if (this.config.bidirectional && this.encCombineProjections) {
			const proj = this.encCombineProjections[this.config.nEncoderLayer - 1]!;
			return proj.forward(encSeq);
		}
		return encSeq;
	}

	forward(
		inputIdsSrc: Tensor,
		inputIdsTgt: Tensor,
		{
			targets,
			srcPaddingMask,
			tgtPaddingMask: _tgtPaddingMask,
			encoderHiddenStates
		}: {
			targets?: Tensor | null;
			srcPaddingMask?: Tensor | null;
			tgtPaddingMask?: Tensor | null;
			encoderHiddenStates?: Tensor | null;
		} = {
			targets: null,
			srcPaddingMask: null,
			tgtPaddingMask: null,
			encoderHiddenStates: null
		}
	): [Tensor, Tensor | null] {
		const encStates = encoderHiddenStates ?? this.encode(inputIdsSrc);
		const y = this.decoderEmbedding.forward(inputIdsTgt); // [B,T,H]
		const attentionOptions: RNNForwardAttentionOptions | undefined = this.attention
			? {
					attentionModule: this.attention,
					encoderOutputs: encStates,
					srcPaddingMask: srcPaddingMask ?? null,
					outProjection: this.outProj,
					inputFeedingProjection: this.decoderInputProj,
					decoderStateProjection: this.decoderStateProj
				}
			: undefined;
		const [hiddenStates] = this.decoder.forward(y, undefined, attentionOptions);
		const logits = this.lmHead.forward(hiddenStates);
		const loss = computeAutoregressiveCrossEntropyLoss(logits, targets ?? null, this.criterion);
		return [logits, loss];
	}
}
