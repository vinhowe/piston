<script lang="ts">
	import { FormLabel, ResetValueButton } from 'example-common';
	import SelectWithCitations from './select/SelectWithCitations.svelte';

	let {
		value = $bindable(),
		id,
		hasDefaultValue = false,
		onReset = undefined
	} = $props<{ value: string; id: string; hasDefaultValue: boolean; onReset: () => void }>();

	// Source: https://www.johndcook.com/blog/2009/01/19/stand-alone-error-function-erf/
	// Abramowitz and Stegun formula 7.1.26
	// Public domain
	function erf(x: number): number {
		// constants
		const a1 = 0.254829592;
		const a2 = -0.284496736;
		const a3 = 1.421413741;
		const a4 = -1.453152027;
		const a5 = 1.061405429;
		const p = 0.3275911;

		// Save the sign of x
		let sign = 1;
		if (x < 0) {
			sign = -1;
		}
		x = Math.abs(x);

		// A&S 7.1.26
		const t = 1.0 / (1.0 + p * x);
		const y = 1.0 - ((((a5 * t + a4) * t + a3) * t + a2) * t + a1) * t * Math.exp(-x * x);

		return sign * y;
	}

	function generatePoints(selectedValue: string) {
		const points = [];
		for (let x = -2.5; x <= 2.5; x += 0.1) {
			let y;
			switch (selectedValue) {
				case 'relu':
					y = Math.max(0, x);
					break;
				case 'relu2':
					y = Math.pow(Math.max(0, x), 2);
					break;
				case 'silu':
					y = x * (1 / (1 + Math.exp(-x)));
					break;
				case 'gelu':
					y = x * 0.5 * (1 + erf(x / Math.sqrt(2)));
					break;
				case 'sigmoid':
					y = 1 / (1 + Math.exp(-x));
					break;
				case 'swiglu':
					y = x * (x * (1 / (1 + Math.exp(-x))));
					break;
				case 'tanh':
					y = Math.tanh(x);
					break;
				default:
					y = x;
			}
			points.push([x, y]);
		}
		return points;
	}

	const points = $derived(generatePoints(value));
	const height = 21;
	const svgPoints = $derived(
		points
			.map(([x, y]) => [
				(x + 2) * (height / 4), // x: [-2,2] -> [0,40]
				height - (y + 2) * (height / 4) // y: [-2,2] -> [40,0]
			])
			.map((point) => point.join(','))
			.join(' ')
	);
</script>

<div class="relative w-full">
	<FormLabel forInputId="activation-function" value="Activation Function" />
	<div class="relative w-full flex items-start gap-1 mt-1">
		<SelectWithCitations
			bind:value
			{id}
			class="w-full"
			options={[
				{
					value: 'relu',
					title: 'ReLU',
					citations: {
						entries: [
							{
								name: 'Nair & Hinton, 2010',
								url: 'https://www.cs.toronto.edu/~fritz/absps/reluICML.pdf'
							}
						]
					}
				},
				{ value: 'relu2', title: 'ReLU²' },
				{
					value: 'gelu',
					title: 'GELU',
					citations: {
						entries: [{ name: 'Hendrycks & Gimpel, 2016', url: 'https://arxiv.org/abs/1606.08415' }]
					}
				},
				{
					value: 'silu',
					title: 'SiLU',
					citations: {
						entries: [
							{
								name: 'Hendrycks & Gimpel, 2016',
								url: 'https://arxiv.org/abs/1606.08415'
							},
							{ name: 'Elfwing et al., 2017', url: 'https://arxiv.org/abs/1702.03118' }
						]
					}
				},
				{
					value: 'sigmoid',
					title: 'Sigmoid',
					citations: {
						entries: [
							{ name: 'Rumelhart et al., 1986', url: 'https://www.nature.com/articles/323533a0' }
						]
					}
				},
				{
					value: 'swiglu',
					title: 'SwiGLU',
					citations: {
						entries: [{ name: 'Shazeer, 2020', url: 'https://arxiv.org/pdf/2002.05202' }]
					}
				},
				{ value: 'tanh', title: 'Tanh' }
			]}
		/>
		<div style={`height: ${height + 2}px`} class="flex items-center gap-1">
			<div class="flex-none">
				<svg width={height + 2} height={height + 2} class="border border-neutral-300 bg-white">
					<!-- Axis lines -->
					<line
						x1="0"
						y1={height / 2}
						x2={height}
						y2={height / 2}
						stroke="#dadada"
						stroke-width="1"
					/>
					<line
						x1={height / 2}
						y1="0"
						x2={height / 2}
						y2={height}
						stroke="#dadada"
						stroke-width="1"
					/>

					<!-- Function line -->
					<polyline points={svgPoints} fill="none" stroke="black" stroke-width="1.5" />
				</svg>
			</div>
			{#if onReset}
				<ResetValueButton {hasDefaultValue} {onReset} />
			{/if}
		</div>
	</div>
</div>
