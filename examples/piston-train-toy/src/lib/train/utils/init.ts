import type {
	Config,
	ProjectionInitializationConfig,
	RNNInitializationConfig,
	XavierInitializationDistribution
} from '$lib/workspace/config';

import {
	initConstant_,
	initNormal_,
	initOnes_,
	initOrthogonal_,
	initXavierNormal_,
	initXavierUniform_,
	initZeros_,
	LayerNorm,
	nn,
	Tensor
} from '@piston-ml/piston-web';
import * as piston from '@piston-ml/piston-web';

import { GRUCell, LSTMCell, RNNCell } from '../model/rnn';

export function initTransformerParameters(self: nn.Module, config: Config): void {
	const initializationConfig = config.model.transformer.initialization;

	if (!initializationConfig.present) {
		return;
	}

	const isEncoderDecoder = config.model.topology === 'encoder-decoder';

	const initTransformerWeights = (module: nn.Module): void => {
		if (module instanceof nn.Linear) {
			initNormal_(module.weight, { mean: 0.0, std: initializationConfig.std });
			if (module.bias != null) {
				initZeros_(module.bias);
			}
		} else if (module instanceof nn.Embedding) {
			initNormal_(module.weight, { mean: 0.0, std: initializationConfig.std });
		} else if (module instanceof nn.LayerNorm) {
			if (module.bias) {
				initZeros_(module.bias);
			}
			initOnes_(module.weight);
		}
	};

	const initProjection = (
		p: nn.Parameter,
		projectionConfig: ProjectionInitializationConfig,
		pn: string
	) => {
		if (!projectionConfig.present) {
			return;
		}
		const nLayers = isEncoderDecoder
			? pn.includes('transformer.encoder')
				? config.model.encoderDecoder.encoderLayers
				: config.model.encoderDecoder.decoderLayers
			: config.model.layers;
		if (projectionConfig.strategy === 'layer-scaled') {
			initNormal_(p, { mean: 0.0, std: 0.02 / Math.sqrt(2 * nLayers) });
		} else if (projectionConfig.strategy === 'zero') {
			initZeros_(p);
		}
	};

	self.apply(initTransformerWeights);
	for (const [pn, p] of self.namedParameters()) {
		// TODO: Make this respect whatever initialization config we have set up
		if (pn.endsWith('cProj.weight')) {
			initProjection(p, initializationConfig.projections.attention, pn);
		}
		if (pn.endsWith('downProj.weight')) {
			initProjection(p, initializationConfig.projections.mlp, pn);
		}
		if (pn.endsWith('lmHead.weight')) {
			initProjection(p, initializationConfig.projections.lmHead, pn);
		}
	}
}

function initInputPart_(
	part: Tensor,
	dist: XavierInitializationDistribution,
	enable: boolean
): Tensor {
	if (!enable) return part;
	return dist === 'uniform' ? initXavierUniform_(part) : initXavierNormal_(part);
}

function initRecurrentParts_(recParts: Tensor[], doOrth: boolean, perGateOrth: boolean): Tensor[] {
	if (!doOrth) return recParts;
	if (perGateOrth) {
		return recParts.map((p) => initOrthogonal_(p));
	} else {
		// One tall orthogonal across all gates (works for rectangular via QR)
		const tall = piston.cat(recParts, { dim: 0 });
		const tallInit = initOrthogonal_(tall);
		return piston.chunk(tallInit, recParts.length, { dim: 0 }) as Tensor[];
	}
}

function initGatedWeight_(
	W: Tensor, // shape: (G*H) x (I+H)
	I: number,
	H: number,
	G: number,
	dist: XavierInitializationDistribution,
	xavierInput: boolean,
	orthRec: boolean,
	perGateOrth: boolean
): Tensor {
	// Split rows into gate blocks
	const gates = piston.chunk(W, G, { dim: 0 }) as Tensor[]; // G × [H x (I+H)]
	// For each gate, split columns into [input | recurrent]
	const inParts: Tensor[] = [];
	const recParts: Tensor[] = [];
	for (const g of gates) {
		const [inp, rec] = piston.split(g, [I, H], { dim: 1 }) as [Tensor, Tensor];
		inParts.push(inp);
		recParts.push(rec);
	}
	const inInit = inParts.map((p) => initInputPart_(p, dist, xavierInput));
	const recInit = initRecurrentParts_(recParts, orthRec, perGateOrth);

	// Reassemble gates, then full matrix
	const gateBlocks = inInit.map((inp, k) => piston.cat([inp, recInit[k]], { dim: 1 }));
	return piston.cat(gateBlocks, { dim: 0 });
}

function buildParameterHacky(tensor: Tensor): Tensor {
	return new nn.Parameter(tensor.debugTensor);
}

const GRU_GATES_COUNT = 2;
const LSTM_GATES_COUNT = 4;

export function initGRUCell_(cell: GRUCell, config: RNNInitializationConfig) {
	const dist = config.xavierInputColumns.distribution;

	const [wgRows, wgCols] = cell.WGates.weight.size();
	const hiddenWG = wgRows / GRU_GATES_COUNT;
	const inputWG = wgCols - hiddenWG;

	// WGates: shape (2H) x (I+H), rows chunked as [r, z]
	cell.WGates.weight = buildParameterHacky(
		initGatedWeight_(
			cell.WGates.weight,
			inputWG,
			hiddenWG,
			GRU_GATES_COUNT,
			dist,
			config.xavierInputColumns.present,
			config.orthogonalRecurrentColumns,
			config.perGateOrthogonalBlocks
		)
	);

	const [wcRows, wcCols] = cell.WCandidate.weight.size();
	const hiddenWCand = wcRows;
	const inputWCand = wcCols - hiddenWCand;
	const [inp, rec] = piston.split(cell.WCandidate.weight, [inputWCand, hiddenWCand], { dim: 1 });
	const inpInit = initInputPart_(inp, dist, config.xavierInputColumns.present);
	const recInit = config.orthogonalRecurrentColumns ? initOrthogonal_(rec) : rec;
	cell.WCandidate.weight = buildParameterHacky(piston.cat([inpInit, recInit], { dim: 1 }));

	const normGates = cell.normGates;

	// Biases
	if (normGates instanceof LayerNorm && normGates.bias) {
		// Keep Linear biases zeroed if requested
		if (cell.WGates.bias)
			cell.WGates.bias = config.zeroBiases
				? buildParameterHacky(initZeros_(cell.WGates.bias))
				: cell.WGates.bias;
		if (cell.WCandidate.bias)
			cell.WCandidate.bias = config.zeroBiases
				? buildParameterHacky(initZeros_(cell.WCandidate.bias))
				: cell.WCandidate.bias;

		// Apply update-gate bias to LN β on z block; r block to zero
		const [beta_r, beta_z] = piston.chunk(normGates.bias, GRU_GATES_COUNT, { dim: 0 });
		const updateGateBias =
			config.gru.updateGateBias?.present && config.gru.updateGateBias.value > 0
				? config.gru.updateGateBias.value
				: 0;
		normGates.bias = buildParameterHacky(
			piston.cat(
				[
					initZeros_(beta_r),
					updateGateBias !== 0 ? initConstant_(beta_z, updateGateBias) : initZeros_(beta_z)
				],
				{ dim: 0 }
			)
		);
	} else {
		// No LN on gates: write into WGates.bias directly (WCandidate bias stays zero if configured)
		if (cell.WGates.bias) {
			const [b_r, b_z] = piston.chunk(cell.WGates.bias, GRU_GATES_COUNT, { dim: 0 });
			const updateGateBias =
				config.gru.updateGateBias?.present && config.gru.updateGateBias.value > 0
					? config.gru.updateGateBias.value
					: 0;

			cell.WGates.bias = buildParameterHacky(
				piston.cat(
					[
						config.zeroBiases ? initZeros_(b_r) : b_r,
						updateGateBias !== 0
							? initConstant_(b_z, updateGateBias)
							: config.zeroBiases
								? initZeros_(b_z)
								: b_z
					],
					{ dim: 0 }
				)
			);
		}
		if (cell.WCandidate.bias) {
			cell.WCandidate.bias = config.zeroBiases
				? buildParameterHacky(initZeros_(cell.WCandidate.bias))
				: cell.WCandidate.bias;
		}
	}
}

export function initLSTMCell_(cell: LSTMCell, config: RNNInitializationConfig) {
	const [wRows, wCols] = cell.W.weight.size();
	const inferredHidden = wRows / LSTM_GATES_COUNT;
	const inferredInput = wCols - inferredHidden;

	cell.W.weight = buildParameterHacky(
		initGatedWeight_(
			cell.W.weight,
			inferredInput,
			inferredHidden,
			LSTM_GATES_COUNT,
			config.xavierInputColumns.distribution,
			config.xavierInputColumns.present,
			config.orthogonalRecurrentColumns,
			config.perGateOrthogonalBlocks
		)
	);

	// Biases/LN handling
	const normGates = cell.normGates;
	const forgetGateBiasPresent = config.lstm.forgetGateBias.present;
	const forgetGateBiasValue = config.lstm.forgetGateBias.value;

	if (normGates instanceof LayerNorm && normGates.bias) {
		// Keep Linear bias zero if requested (recommended with LN)
		if (cell.W.bias)
			cell.W.bias = config.zeroBiases ? buildParameterHacky(initZeros_(cell.W.bias)) : cell.W.bias;

		// LN beta blocks: [i, f, g, o]; set f to forgetVal, others to 0
		const [b_i, b_f, b_g, b_o] = piston.chunk(normGates.bias, LSTM_GATES_COUNT, {
			dim: 0
		});
		normGates.bias = buildParameterHacky(
			piston.cat(
				[
					initZeros_(b_i),
					forgetGateBiasPresent ? initConstant_(b_f, forgetGateBiasValue) : initZeros_(b_f),
					initZeros_(b_g),
					initZeros_(b_o)
				],
				{ dim: 0 }
			)
		);
	} else if (cell.W.bias) {
		const [b_i, b_f, b_g, b_o] = piston.chunk(cell.W.bias, LSTM_GATES_COUNT, {
			dim: 0
		});
		cell.W.bias = buildParameterHacky(
			piston.cat(
				[
					config.zeroBiases ? initZeros_(b_i) : b_i,
					forgetGateBiasPresent
						? initConstant_(b_f, forgetGateBiasValue)
						: config.zeroBiases
							? initZeros_(b_f)
							: b_f,
					config.zeroBiases ? initZeros_(b_g) : b_g,
					config.zeroBiases ? initZeros_(b_o) : b_o
				],
				{ dim: 0 }
			)
		);
	}
}

export function initRNNCell_(cell: RNNCell, config: RNNInitializationConfig) {
	if (config.xavierInputColumns.present) {
		const dist = config.xavierInputColumns.distribution;
		cell.WIh.weight = buildParameterHacky(
			dist === 'uniform' ? initXavierUniform_(cell.WIh.weight) : initXavierNormal_(cell.WIh.weight)
		);
	}

	if (cell.WIh.bias) {
		cell.WIh.bias = config.zeroBiases
			? buildParameterHacky(initZeros_(cell.WIh.bias))
			: cell.WIh.bias;
	}

	if (config.orthogonalRecurrentColumns) {
		cell.WHh.weight = buildParameterHacky(initOrthogonal_(cell.WHh.weight));
	}

	if (cell.WHh.bias) {
		cell.WHh.bias = config.zeroBiases
			? buildParameterHacky(initZeros_(cell.WHh.bias))
			: cell.WHh.bias;
	}
}

export function initRNNParameters(root: nn.Module, config: Config) {
	if (!config.model.rnn.initialization.present) return;

	const initializeRNNCell = (module: nn.Module) => {
		if (module instanceof GRUCell) {
			initGRUCell_(module, config.model.rnn.initialization);
		} else if (module instanceof LSTMCell) {
			initLSTMCell_(module, config.model.rnn.initialization);
		} else if (module instanceof RNNCell) {
			initRNNCell_(module, config.model.rnn.initialization);
		}
	};

	root.apply(initializeRNNCell);
}
