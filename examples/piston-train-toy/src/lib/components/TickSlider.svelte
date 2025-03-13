<script lang="ts">
	import BaseSlider from './BaseSlider.svelte';

	export let value: number;
	export let min: number;
	export let max: number;
	export let step: number = 1;
	export let label: string;
	export let formatter: (value: number) => string = (v) => v.toString();

	// Normalize the slider bounds
	$: low = Math.min(min, max);
	$: high = Math.max(min, max);

	// Generate tick marks based on the normalized range.
	function generateTicks() {
		if (high === low) return [];
		const ticks = [];
		const range = high - low;
		const totalTicks = range / step;
		const majorTickCount = 10; // Show 10 major ticks
		const minorTickCount = 40; // Show at most 40 minor ticks

		const majorStep = Math.ceil(totalTicks / majorTickCount) * step;
		const minorStep = Math.ceil(totalTicks / minorTickCount) * step;

		// Add major ticks
		for (let i = low; i <= high; i += majorStep) {
			ticks.push({
				value: i,
				label: formatter(i),
				type: 'major'
			});
		}

		// Add minor ticks (only if not very close to a major tick)
		for (let i = low; i <= high; i += minorStep) {
			if (!ticks.some((t) => Math.abs(t.value - i) < step / 2)) {
				ticks.push({
					value: i,
					label: '',
					type: 'minor'
				});
			}
		}

		return ticks.sort((a, b) => a.value - b.value);
	}

	// Regenerate ticks reactively when low, high, step, or formatter change.
	$: ticks = generateTicks();

	// Compute tick positions reactively so they update when low/high change.
	$: tickPositions = ticks.map((tick) => ({
		...tick,
		position: `${((tick.value - low) / (high - low)) * 100}%`
	}));

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

	<div slot="ticks" class="ticks-container">
		{#each tickPositions as tick}
			<div class="slider-tick" style:left={tick.position}>
				<div class="slider-tick-mark {tick.type}"></div>
			</div>
		{/each}
	</div>

	<input
		slot="input"
		type="range"
		min={low}
		max={high}
		{step}
		bind:value
		on:input={handleSliderChange}
		class="slider-input"
	/>
</BaseSlider>

<style>
	/* Ensure the ticks container is positioned relative so that absolutely-
	   positioned ticks are aligned relative to it. */
	.ticks-container {
		position: relative;
		width: 100%;
	}

	/* Tick elements are absolutely positioned and centered on their computed left coordinate. */
	.slider-tick {
		position: absolute;
		top: 0;
		transform: translateX(-50%);
	}
</style>
