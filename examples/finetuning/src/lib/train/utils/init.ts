// import type { Config, ProjectionInitializationConfig } from '$lib/workspace/config';

// import { initNormal_, initOnes_, initZeros_, nn } from '@piston-ml/piston-web';

// TODO: Setup initialization for GPT2 lora

// export function initTransformerParameters(self: nn.Module, config: Config): void {
// 	const initializationConfig = config.model.transformer.initialization;

// 	if (!initializationConfig.present) {
// 		return;
// 	}

// 	const initTransformerWeights = (module: nn.Module): void => {
// 		if (module instanceof nn.Linear) {
// 			initNormal_(module.weight, { mean: 0.0, std: initializationConfig.std });
// 			if (module.bias != null) {
// 				initZeros_(module.bias);
// 			}
// 		} else if (module instanceof nn.Embedding) {
// 			initNormal_(module.weight, { mean: 0.0, std: initializationConfig.std });
// 		} else if (module instanceof nn.LayerNorm) {
// 			if (module.bias) {
// 				initZeros_(module.bias);
// 			}
// 			initOnes_(module.weight);
// 		}
// 	};

// 	const initProjection = (p: nn.Parameter, projectionConfig: ProjectionInitializationConfig) => {
// 		if (!projectionConfig.present) {
// 			return;
// 		}
// 		const nLayers = config.model.layers;
// 		if (projectionConfig.strategy === 'layer-scaled') {
// 			initNormal_(p, { mean: 0.0, std: 0.02 / Math.sqrt(2 * nLayers) });
// 		} else if (projectionConfig.strategy === 'zero') {
// 			initZeros_(p);
// 		}
// 	};

// 	self.apply(initTransformerWeights);
// 	for (const [pn, p] of self.namedParameters()) {
// 		if (pn.endsWith('cProj.weight')) {
// 			initProjection(p, initializationConfig.projections.attention);
// 		}
// 		if (pn.endsWith('downProj.weight')) {
// 			initProjection(p, initializationConfig.projections.mlp);
// 		}
// 		if (pn.endsWith('lmHead.weight')) {
// 			initProjection(p, initializationConfig.projections.lmHead);
// 		}
// 	}
// }
