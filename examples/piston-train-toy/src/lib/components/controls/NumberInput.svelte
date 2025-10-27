<script lang="ts">
	import FormLabel from './FormLabel.svelte';
	import ResetValueButton from './ResetValueButton.svelte';

	type $$Props = {
		label?: string;
		value: number;
		id: string;
		step?: number;
		min?: number;
		max?: number;
		unit?: string;
		class?: string;
		hasDefaultValue?: boolean;
		onReset?: () => void;
	};

	let {
		label,
		value = $bindable(),
		id,
		step,
		min,
		max,
		unit,
		class: wrapperClass = '',
		hasDefaultValue = false,
		onReset = undefined
	}: $$Props = $props();
</script>

<div class={wrapperClass}>
	{#if label}
		<FormLabel forInputId={id} value={label} />
	{/if}
	<div class="flex gap-1.5 items-center">
		{#if unit}
			<div class="mt-1 flex">
				<input
					type="number"
					{id}
					bind:value
					{step}
					{min}
					{max}
					class="block w-full border border-panel-border-base px-1.5 py-0.5 focus:border-neutral-500 focus:outline-none text-controls-numeric font-mono"
				/>
				<span
					class="inline-flex items-center border border-l-0 border-panel-border-base bg-neutral-50 px-3 text-neutral-500"
				>
					{unit}
				</span>
			</div>
		{:else}
			<input
				type="number"
				{id}
				bind:value
				{step}
				{min}
				{max}
				class="mt-1 block w-full border border-panel-border-base px-1.5 py-0.5 focus:border-neutral-500 focus:outline-none text-controls-numeric font-mono"
			/>
		{/if}
		{#if onReset}
			<div class="translate-y-0.5">
				<ResetValueButton {hasDefaultValue} {onReset} />
			</div>
		{/if}
	</div>
</div>
