import type { Config } from './config';

export type VisualizationExample = {
	label: string;
	script: string;
	predicate?: (config: Config) => boolean;
};

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

const TUTORIAL = `// Tutorial Introduction to Capture Query Language
//
// - These examples assume you're training a basic
//   transformer model; you'll need to adjust them for other
//   architectures.
// - Examples are commented out to avoid cluttering the
//   canvas.
// - Toggle comments with Ctrl/Cmd + /, and click the green
//   play button to apply the example.
// - If the interpreter can't parse the example, you'll see
//   an error lint.

//
// MODULE SELECTION
//

// This implicitly selects the tensor output of the first
// layer, index 0 of a ModuleList called \`layer\`. In this
// case, the output is likely a list, so all tensors in that
// list will be rendered.
//
layer[0]

// The above is equivalent to:
//
// layer[0] :output

// :output is a type of specifier that changes the query in
// some way. Another example is :grad(ient):
//
// layer[0] :grad

// In addition to selecting modules by name, we can select
// them by type. For example, to select the output of all
// modules with the class name SelfAttention, we can use the
// following query:
//
// .SelfAttention

// What if we want to select all modules with Attention in
// the name? We can use a type regex:
//
// layer[0] ./.*Attention/

// We can also select modules by name regex:
//
// /c(Proj|Attn)/

// Additionally, we can use wildcards in queries. For
// example, to select the output of all submodules of the
// first layer, we can use the following query:
//
// layer[0] *

// We can select more specific modules by using a selector
// chain. For example, to select all attention modules in
// the first layer, we can use the following query:
//
// layer[0] ./.*Attention/

// That will select all children, immediately or not, that
// match the regex. We can also select immediate children
// only by using the child combinator...
//
// layer[0] > ./.*Attention/

// ...or all next modules in registration order which we try
// to keep pretty close to execution order)...
//
// layer[0] .SelfAttention ~ .RMSNorm

// ...or all subsequent modules in registration order.
//
// .RMSNorm + .Embedding

//
// OP SELECTION
//

// A model's forward pass is composed of a series of math
// functions on tensor inputs, which we call operations, or
// ops. We can look at the output of a particular operation
// by name. To see all ReLU² activations in the first layer,
// we can use an op selector, which is denoted by the @
// symbol:
//
// layer[0] * @ Relu2

// We can do next- and subsequent-sibling selectors on
// operators too:
//
// layer[0] ./.*Attention/ @ WhereCond + Softmax

// Wildcards as well:
//
// layer[0] @ *

//
// PARAMETER SELECTION
//

// So far, we've only looked at model activations. We might
// also like to see how parameters change over time. We can
// do this by using a parameter selector, which is denoted
// by the # symbol:
//
// layer[0] * # weight

// Parameter selectors also support regexes and wildcards:
//
// layer[0] * # /(weight|bias)/
// layer[0] * # *

//
// PIPING
//

// We can also apply simple JavaScript to the output of a
// query. For example, it's common that the input or output
// of a module is an object, like a list, containing
// tensors. We can pipe the output to a JavaScript function
// to transform it. For example, to get the first tensor
// from the output of the first layer, we can use the
// following query:
//
// layer[0] :input |{return it[0]}

// We can also apply arbitrary tensor operations to the
// output of a query. For example, to get only the positive
// values of the input of the first layer, we can use the
// following query:
//
// layer[0] :input |{return it[0].where(it[0].ge(0), 0)}

//
// INDEXING
//

// We can slice the tensor output of a query. For example,
// to get the first two heads of the attention probabilities
// of the first layer, we can use the following query, which
// should feel familiar if you've used NumPy:
//
// layer[0] ./.*Attention/ @ WhereCond + Softmax [:,:2]

// We also support start:stop and ellipsis, though the
// practical value of this particular example is limited.
//
// layer[0] ./.*Attention/ @ WhereCond + Softmax [...,0:2]

//
// VISUALIZATION UTILITIES
//

// Often the rendered size of the tensor is either smaller
// or larger than desired. We can use the :scale(x) modifier
// to scale the tensor by a factor of x.
//
// layer[0] ./.*Attention/ @ WhereCond + Softmax :scale(5)

// Equivalently, we support :scale(x%) to scale the tensor
// by a factor of x%.
//
// layer[0] ./.*Attention/ @ WhereCond + Softmax :scale(500%)

// We can also use the :label("label") modifier to add a
// label to the tensor.
//
// layer[0] ./.*Attention/ @ WhereCond + Softmax :scale(5) :label("attention probabilities $\\sigma$")`;

const ATTENTION_PROBABILITIES = `./.*Attention/ @ Softmax :label("attention probabilities") :scale(5)`;
const ATTENTION_ACTIVATIONS = `layer[0] ./.*Attention/ @ * :label("attention activations") :scale(2)`;
const ATTENTION_GRADIENTS = `layer[0] ./.*Attention/ @ * :grad :label("attention gradients") :scale(2)`;
const ATTENTION_PARAMETERS = `layer[0] ./.*Attention/ * #* :label("attention weights (descendants)") :scale(2)
layer[0] ./.*Attention/ #* :label("attention weights") :scale(2)`;

const MLP_ACTIVATIONS = `layer[0] .MLP @ * :label("mlp activations") :scale(2)`;

const ATTENTION_PREDICATE = (config: Config) => {
	return config.model.family === 'transformer';
};

export const VISUALIZATION_EXAMPLES: Record<string, VisualizationExample> = {
	'attention-activations': {
		label: 'Attention Activations',
		script: ATTENTION_ACTIVATIONS,
		predicate: ATTENTION_PREDICATE
	},
	tutorial: { label: 'Tutorial Introduction', script: TUTORIAL },
	'attention-probabilities': {
		label: 'Attention Probabilities',
		script: ATTENTION_PROBABILITIES,
		predicate: ATTENTION_PREDICATE
	},
	'attention-gradients': {
		label: 'Attention Gradients',
		script: ATTENTION_GRADIENTS,
		predicate: ATTENTION_PREDICATE
	},
	'attention-parameters': {
		label: 'Attention Parameters',
		script: ATTENTION_PARAMETERS,
		predicate: ATTENTION_PREDICATE
	},
	'mlp activations': {
		label: 'MLP Activations',
		script: MLP_ACTIVATIONS,
		predicate: ATTENTION_PREDICATE
	},
	'kitchen-sink': { label: 'Kitchen Sink', script: KITCHEN_SINK, predicate: ATTENTION_PREDICATE },
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
	const baseOptions = [
		...Object.entries(VISUALIZATION_EXAMPLES)
			.filter(([_, e]) => (e.predicate ? e.predicate(config) : true))
			.map(([id, e]) => ({ value: id, text: e.label }))
	];

	const customOption = { value: 'custom', text: 'Custom' };

	if (config.visualization.example === 'custom') {
		// We want to show it first in this case
		return [customOption, ...baseOptions];
	}
	return [...baseOptions, customOption];
}
