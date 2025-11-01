import type { Config } from './config';

export type VisualizationExample = {
	label: string;
	script: string;
	predicate?: (config: Config) => boolean;
};

// Kitchen Sink example content taken from previous default, kept verbatim
const KITCHEN_SINK = `// Kitchen-sink Capture Query Language example

//
// Retained gradient of first two heads of attention probabilities:
//
// 1. index 0 in the in the ModuleList named \`layer\`—the first decoder layer…
// 2. …with any descendant where the module class name matches the regex ./.*Attention/ (note the
// 		preceding \`.\`), not necessarily immediately…
// 3. …\`@\` ends the module selector and begins selecting operations…
// 4. …with Softmax immediately following WhereCond…
// 5. …where we call :grad(ient) to get the gradient…
// 6. …labeling the result with :label("Self-Attention Gradients")…
// 7. …using :scale(5) to scale the result by 5x…
// 8. …and we only want the first two heads, skipping the batch dimension.
layer[0] ./.*Attention/ @ WhereCond + Softmax :grad :label("self-attention $\\sigma$ gradients") :scale(5) [:,:2]

//
// Positive values of input of every MLP module:
//
// 1. .MLP selects all modules with the class name \`MLP\`.
// 2. We can filter the captured value with JavaScript. In this case, the input
//    to the module is a list of arguments, so we return the first argument before
//    selecting for positive values. The input is passed in as the value \`it\`:
.MLP :input :scale(3) :label("mlp input ≥ 0") |{
  const tensor = it[0];
  return tensor.where(tensor.ge(0), 0);
}

//
// Wildcard selector for every bias parameter
//
//
* #bias :label("all biases") :scale(3)`;

const ATTENTION_PROBABILITIES = `./.*Attention/ @ Softmax :label("attention probabilities") :scale(5)`;
const ATTENTION_ACTIVATIONS = `layer[0] ./.*Attention/ @ * :label("attention activations") :scale(2)`;
const ATTENTION_GRADIENTS = `layer[0] ./.*Attention/ @ * :grad :label("attention gradients") :scale(2)`;
const ATTENTION_PARAMETERS = `layer[0] ./.*Attention/ * #* :label("attention weights (descendants)") :scale(2)
layer[0] ./.*Attention/ #* :label("attention weights") :scale(2)`;

const MLP_ACTIVATIONS = `layer[0] .MLP @ * :label("mlp activations") :scale(2)`;

export const VISUALIZATION_EXAMPLES: Record<string, VisualizationExample> = {
	'attention-probabilities': {
		label: 'Attention Probabilities',
		script: ATTENTION_PROBABILITIES
	},
	'attention-activations': {
		label: 'Attention Activations',
		script: ATTENTION_ACTIVATIONS
	},
	'attention-gradients': {
		label: 'Attention Gradients',
		script: ATTENTION_GRADIENTS
	},
	'attention-parameters': {
		label: 'Attention Parameters',
		script: ATTENTION_PARAMETERS
	},
	'mlp activations': {
		label: 'MLP Activations',
		script: MLP_ACTIVATIONS
	},
	'kitchen-sink': { label: 'Kitchen Sink', script: KITCHEN_SINK },
	'all-activations': {
		label: 'All Activations',
		script: '* @ * :scale(3)'
	},
	'all-gradients': {
		label: 'All Gradients',
		script: '* @ * :grad :scale(3)'
	},
	'all-parameters': {
		label: 'All Parameters',
		script: '* # * :scale(3)'
	}
};

export const VISUALIZATION_EXAMPLES_BY_SCRIPT = new Map(
	Object.entries(VISUALIZATION_EXAMPLES).map(([id, example]) => [
		example.script,
		{ ...example, id }
	])
);

const DEFAULT_VISUALIZATION_EXAMPLE = VISUALIZATION_EXAMPLES['attention-probabilities'];

export function getVisualizationExampleById(id: string | null | undefined): VisualizationExample {
	if (!id) return DEFAULT_VISUALIZATION_EXAMPLE;
	return VISUALIZATION_EXAMPLES[id] ?? DEFAULT_VISUALIZATION_EXAMPLE;
}

export function findExampleIdMatchingScript(script: string | null | undefined): string | null {
	if (script == null) return null;
	const match = VISUALIZATION_EXAMPLES_BY_SCRIPT.get(script);
	return match ? match.id : null;
}

export function getEffectiveVisualizationScript(
	exampleId: string | null | undefined,
	customScript: string | null | undefined
): string {
	return exampleId === 'custom'
		? (customScript ?? '')
		: getVisualizationExampleById(exampleId).script;
}

export function getVisualizationExampleOptions(
	config: Config
): Array<{ value: string; text: string }> {
	return [
		...Object.entries(VISUALIZATION_EXAMPLES).map(([id, e]) => ({ value: id, text: e.label })),
		{ value: 'custom', text: 'Custom' }
	];
}
