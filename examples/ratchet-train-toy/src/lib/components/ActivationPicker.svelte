<script lang="ts">
	import { create, all } from 'mathjs';
	const math = create(all);

	let { value = $bindable() } = $props<{value: string}>();

	function generatePoints(selectedValue: string) {
		const points = [];
		for (let x = -2; x <= 2; x += 0.1) {
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
					y = x * 0.5 * (1 + math.erf(x / Math.sqrt(2)));
					break;
				case 'sigmoid':
					y = 1 / (1 + Math.exp(-x));
					break;
				case 'swiglu':
					y = x * (x * (1 / (1 + Math.exp(-x))));
					break;
				default:
					y = x;
			}
			points.push([x, y]);
		}
		return points;
	}

	const points = $derived(generatePoints(value));
	const svgPoints = $derived(
		points
			.map(([x, y]) => [
				(x + 2) * 10, // x: [-2,2] -> [0,40]
				40 - (y + 2) * 10 // y: [-2,2] -> [40,0]
			])
			.map((point) => point.join(','))
			.join(' ')
	);
</script>

<div class="flex items-center gap-1">
	<div class="relative flex-1">
		<select
			bind:value
			class="w-full p-2 pr-8 border focus:outline-none focus:border-gray-400 border-gray-400 bg-white appearance-none"
		>
			<option value="relu">ReLU</option>
			<option value="relu2">ReLU&sup2;</option>
			<option value="gelu">GELU</option>
			<option value="silu">SiLU</option>
			<option value="sigmoid">Sigmoid</option>
			<option value="swiglu">SwiGLU</option>
		</select>
		<div class="pointer-events-none absolute inset-y-0 right-0 flex items-center px-2 text-gray-400">
			<svg class="h-4 w-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"
				><path d="M19 9l-7 7-7-7" stroke-linecap="round" stroke-linejoin="round" stroke-width="2" /></svg
			>
		</div>
	</div>
	<div class="flex-none">
		<svg width="42" height="42" class="border border-gray-400 bg-white">
			<!-- Axis lines -->
			<line x1="0" y1="20" x2="40" y2="20" stroke="#dadada" stroke-width="1" />
			<line x1="20" y1="0" x2="20" y2="40" stroke="#dadada" stroke-width="1" />

			<!-- Function line -->
			<polyline points={svgPoints} fill="none" stroke="black" stroke-width="1.5" />
		</svg>
	</div>
</div> 