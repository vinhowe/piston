<script lang="ts">
	import BaseSlider from './BaseSlider.svelte';

	export let value: number;
	export let min: number;
	export let max: number;
	export let step: number = 1;
	export let label: string;
	export let formatter: (value: number) => string = (v) => v.toString();
</script>

<BaseSlider {label} {value}>
	<div slot="value-display" class="text-center mb-2 font-mono text-sm">
		{formatter(value)}
	</div>
	<input
		slot="input"
		type="range"
		{min}
		{max}
		{step}
		bind:value
		class="slider-input"
	/>
</BaseSlider> 