import type { OptimizerConfig } from '$lib/workspace/config';

import {
	AdamW,
	type Device,
	type Module,
	nn,
	type Optimizer,
	type Parameter as ParameterType,
	type ParamGroup,
	SGD
} from '@piston-ml/piston-web';

function paramsFromNamesSorted(
	names: Iterable<string>,
	paramDict: Map<string, ParameterType>
): ParameterType[] {
	return Array.from(names)
		.sort()
		.map((name) => paramDict.get(name)!)
		.filter((p) => p != null);
}

/**
 * Validates that all model parameters are included in the parameter groups.
 * Throws an error if any parameters are missing from the groups.
 * @param model - The model to validate
 * @param paramGroups - The parameter groups to check
 * @throws Error if any model parameters are not included in the parameter groups
 */
function validateParameterGroups(
	model: Module,
	paramGroups: ParamGroup[],
	paramDict: Map<string, ParameterType>
): void {
	// Get all parameters from the model
	const allModelParams = new Set<ParameterType>();
	for (const [_, param] of model.namedParameters()) {
		allModelParams.add(param);
	}

	// Get all parameters from the parameter groups
	const groupParams = new Set<ParameterType>();
	for (const group of paramGroups) {
		for (const param of group.params) {
			groupParams.add(param);
		}
	}

	// Find parameters that are in the model but not in any group
	const missingParams: ParameterType[] = [];
	for (const param of allModelParams) {
		if (!groupParams.has(param)) {
			missingParams.push(param);
		}
	}

	if (missingParams.length > 0) {
		// Find the names of the missing parameters using paramDict
		const missingParamNames: string[] = [];
		for (const [name, param] of paramDict) {
			if (missingParams.includes(param)) {
				missingParamNames.push(name);
			}
		}

		throw new Error(
			`Found ${missingParams.length} model parameters that are not included in any parameter group (${groupParams.size} included). ` +
				`All model parameters must be assigned to a parameter group for training. ` +
				`Missing parameters: ${missingParamNames.join(', ')}`
		);
	}
}

function getWeightDecayParams(
	model: Module,
	useWeightDecayGroups: boolean,
	// eslint-disable-next-line @typescript-eslint/no-explicit-any
	whitelistWeightModules: (new (...args: any[]) => Module<any, any>)[],
	// eslint-disable-next-line @typescript-eslint/no-explicit-any
	blacklistWeightModules: (new (...args: any[]) => Module<any, any>)[]
): { paramDict: Map<string, ParameterType>; decay: Set<string>; noDecay: Set<string> } {
	const decay = new Set<string>();
	const noDecay = new Set<string>();

	const paramDict = new Map<string, ParameterType>();

	for (const [mn, m] of model.namedModules()) {
		for (const [pn, p] of m.namedParameters()) {
			const fpn = mn ? `${mn}.${pn}` : pn;

			paramDict.set(fpn, p);

			if (useWeightDecayGroups) {
				if (pn.endsWith('bias')) {
					// All biases will not be decayed
					noDecay.add(fpn);
				} else if (pn.endsWith('weight')) {
					if (whitelistWeightModules.some((cls) => m instanceof cls)) {
						// Weights of whitelist modules will be weight decayed
						decay.add(fpn);
					} else if (blacklistWeightModules.some((cls) => m instanceof cls)) {
						// Weights of blacklist modules will NOT be weight decayed
						noDecay.add(fpn);
					}
				} else {
					// Parameters that are not weights or biases (shouldn't exist in std models)
					// Add to decay by default, adjust if necessary for specific models.
					decay.add(fpn);
				}
			} else {
				decay.add(fpn);
			}
		}
	}

	if (useWeightDecayGroups) {
		// Validate that we considered every parameter
		const allParamNames = new Set<string>(
			Array.from(model.namedParameters()).map(([name]) => name)
		);
		const interParams = new Set([...decay].filter((x) => noDecay.has(x)));
		const unionParams = new Set([...decay, ...noDecay]);

		if (interParams.size !== 0) {
			throw new Error(
				`Parameters ${JSON.stringify(Array.from(interParams))} made it into both decay/noDecay sets`
			);
		}
		const missingParams = new Set([...allParamNames].filter((x) => !unionParams.has(x)));
		if (missingParams.size !== 0) {
			throw new Error(
				`Parameters ${JSON.stringify(
					Array.from(missingParams)
				)} were not separated into either decay/noDecay set`
			);
		}
	}

	return { paramDict, decay, noDecay };
}

/**
 * Based on what minGPT does:
 * Configures the optimizer based on the training configuration.
 * Separates parameters into weight decay and no weight decay groups.
 * @param trainConfig - The optimizer
 * configuration
 * @param device - The computation device
 * @returns The configured optimizer
 */
function configureOptimizers(
	model: Module,
	moduleLayersPrefixes: string[],
	lmHeadPrefix: string,
	trainConfig: OptimizerConfig,
	device: Device
): Optimizer {
	const whitelistWeightModules = [nn.Linear];
	const blacklistWeightModules = [nn.LayerNorm, nn.RMSNorm, nn.Embedding];

	const effectiveWeightDecay = trainConfig.weightDecay.present
		? trainConfig.weightDecay.value
		: 0.0;

	const { paramDict, decay, noDecay } = getWeightDecayParams(
		model,
		trainConfig.weightDecay.useWeightDecayGroups,
		whitelistWeightModules,
		blacklistWeightModules
	);
	// Deterministic param lists by name
	const decayParamsValues = paramsFromNamesSorted(decay, paramDict);
	const noDecayParamsValues = paramsFromNamesSorted(noDecay, paramDict);

	if (trainConfig.type === 'AdamW' || trainConfig.type === 'Adam' || trainConfig.type === 'SGD') {
		const optimGroups: ParamGroup[] = [
			{
				params: decayParamsValues,
				weightDecay: effectiveWeightDecay
			},
			...(noDecayParamsValues.length > 0
				? [
						{
							params: noDecayParamsValues,
							weightDecay: 0.0 // no decay
						}
					]
				: [])
		];

		validateParameterGroups(model, optimGroups, paramDict);

		// Create the AdamW optimizer
		if (trainConfig.type === 'AdamW' || trainConfig.type === 'Adam') {
			return new AdamW(optimGroups, device, {
				lr: trainConfig.lr,
				betas: [trainConfig.adam.beta1, trainConfig.adam.beta2],
				eps: trainConfig.adam.eps,
				weightDecay: effectiveWeightDecay,
				amsgrad: trainConfig.adam.amsgrad
			});
		} else if (trainConfig.type === 'SGD') {
			return new SGD(optimGroups, device, {
				lr: trainConfig.lr,
				momentum: trainConfig.sgd.momentum,
				dampening: trainConfig.sgd.dampening,
				weightDecay: effectiveWeightDecay,
				nesterov: trainConfig.sgd.nesterov
			});
		}
	}

	throw new Error(`Unknown optimizer type: ${trainConfig.type}`);
}

export function configureOptimizerForModel(
	model: Module,
	isEncoderOnly: boolean,
	isEncoderDecoder: boolean,
	config: OptimizerConfig,
	device: Device
): Optimizer {
	if (isEncoderOnly) {
		return configureOptimizers(model, ['encoder.layer'], 'mlmHead', config, device);
	} else if (isEncoderDecoder) {
		return configureOptimizers(model, ['decoder.layer', 'encoder.layer'], 'lmHead', config, device);
	} else {
		return configureOptimizers(model, ['decoder.layer'], 'lmHead', config, device);
	}
}
