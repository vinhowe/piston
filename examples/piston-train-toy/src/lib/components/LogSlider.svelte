<script lang="ts">
	import BaseSlider from './BaseSlider.svelte';

	export let value: number;
	export let minExp: number;
	export let maxExp: number;
	export let label: string;

	// Generate all snap points
	function generateSnapPoints() {
		const points = new Set<number>();
		for (let exp = minExp; exp <= maxExp; exp++) {
			// Add power of 10
			points.add(exp);

			// Add 2x and 5x multiples
			if (exp < maxExp) {
				points.add(exp + Math.log10(2));
				points.add(exp + Math.log10(5));
			}
		}
		return Array.from(points).sort((a, b) => a - b);
	}

	const snapPoints = generateSnapPoints();

	// Find nearest snap point
	function findNearestSnapPoint(val: number) {
		// Fixed small tolerance for more precise snapping
		const tolerance = 0.05;

		let nearest = val;
		let minDist = tolerance;

		for (const point of snapPoints) {
			const dist = Math.abs(val - point);
			if (dist < minDist) {
				minDist = dist;
				nearest = point;
			}
		}

		return minDist < tolerance ? nearest : val;
	}

	// Generate tick marks
	function generateTicks() {
		const ticks = [];
		for (let i = minExp; i <= maxExp; i++) {
			// Major ticks (powers of 10)
			ticks.push({
				value: i,
				label: i === 0 ? '1' : `10<sup>${i}</sup>`,
				type: 'major'
			});

			// Minor ticks
			if (i < maxExp) {
				// Add 2x and 5x ticks
				ticks.push({
					value: i + Math.log10(2),
					label: '',
					type: 'medium'
				});
				ticks.push({
					value: i + Math.log10(5),
					label: '',
					type: 'medium'
				});

				// Additional minor ticks
				[3, 4, 6, 7, 8, 9].forEach((n) => {
					ticks.push({
						value: i + Math.log10(n),
						label: '',
						type: 'minor'
					});
				});
			}
		}
		return ticks.sort((a, b) => a.value - b.value);
	}

	const ticks = generateTicks();
	$: logValue = Math.pow(10, value);

	function formatNumber(num: number) {
		return num.toExponential(2);
	}

	function handleSliderChange(e: Event) {
		const rawValue = parseFloat((e.target as HTMLInputElement).value);
		value = findNearestSnapPoint(rawValue);
	}
</script>

<BaseSlider {label} {value}>
	<div slot="value-display" class="text-center mb-2 font-mono text-sm">
		{formatNumber(logValue)}
	</div>

	<div slot="ticks">
		{#each ticks as tick}
			{@const position = `${((tick.value - minExp) / (maxExp - minExp)) * 100}%`}
			{@const isSnapPoint = snapPoints.includes(tick.value)}
			<div class="slider-tick" style:left={position}>
				<div class="slider-tick-mark {tick.type} {isSnapPoint ? 'snap' : ''}" />
				{#if tick.type === 'major'}
					<div class="slider-tick-label">
						{@html tick.label}
					</div>
				{/if}
			</div>
		{/each}
	</div>

	<input
		slot="input"
		type="range"
		min={minExp}
		max={maxExp}
		step={0.001}
		bind:value
		on:input={handleSliderChange}
		class="slider-input"
	/>
</BaseSlider> 