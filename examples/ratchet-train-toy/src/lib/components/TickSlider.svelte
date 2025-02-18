<script lang="ts">
	import BaseSlider from './BaseSlider.svelte';

	export let value: number;
	export let min: number;
	export let max: number;
	export let step: number = 1;
	export let label: string;
	export let formatter: (value: number) => string = (v) => v.toString();

	// Generate tick marks
	function generateTicks() {
		const ticks = [];
		const range = max - min;
		const totalTicks = range / step;
		const majorTickCount = 10; // Show 10 major ticks
		const minorTickCount = 40; // Show at most 20 minor ticks
		
		const majorStep = Math.ceil(totalTicks / majorTickCount) * step;
		const minorStep = Math.ceil(totalTicks / minorTickCount) * step;

		// Add major ticks
		for (let i = min; i <= max; i += majorStep) {
			ticks.push({
				value: i,
				label: formatter(i),
				type: 'major'
			});
		}

		// Add minor ticks
		for (let i = min; i <= max; i += minorStep) {
			if (!ticks.some(t => Math.abs(t.value - i) < step/2)) {
				ticks.push({
					value: i,
					label: '',
					type: 'minor'
				});
			}
		}

		return ticks.sort((a, b) => a.value - b.value);
	}

	const ticks = generateTicks();

	function handleSliderChange(e: Event) {
		const rawValue = parseFloat((e.target as HTMLInputElement).value);
		// Snap to nearest step
		value = Math.round(rawValue / step) * step;
	}
</script>

<BaseSlider {label} {value}>
	<div slot="value-display" class="text-center mb-2 font-mono text-sm">
		{formatter(value)}
	</div>

	<div slot="ticks">
		{#each ticks as tick}
			{@const position = `${((tick.value - min) / (max - min)) * 100}%`}
			<div class="slider-tick" style:left={position}>
				<div class="slider-tick-mark {tick.type}" />
			</div>
		{/each}
	</div>

	<input
		slot="input"
		type="range"
		{min}
		{max}
		{step}
		bind:value
		on:input={handleSliderChange}
		class="slider-input"
	/>
</BaseSlider> 