<script lang="ts">
	import FormLabel from './FormLabel.svelte';
	import ResetValueButton from './ResetValueButton.svelte';

	type $$Props = {
		label?: string;
		value: string | number;
		id: string;
		class?: string;
		placeholder?: string;
		type?: string;
		hasDefaultValue?: boolean;
		onReset?: () => void;
	};

	let {
		label,
		value = $bindable(),
		id,
		class: wrapperClass = '',
		placeholder = '',
		type = 'text',
		hasDefaultValue = false,
		onReset = undefined
	}: $$Props = $props();
</script>

<div class="relative {wrapperClass}">
	{#if label}
		<FormLabel forInputId={id} value={label} />
	{/if}
	<div class="flex gap-1.5 items-center">
		<input
			{id}
			{type}
			{placeholder}
			bind:value
			class="block w-full border border-neutral-300 py-0.5 px-1 text-base focus:border-blue-500 focus:outline-none focus:ring-blue-500 sm:text-base"
			class:mt-1={label}
		/>
		{#if onReset}
			<ResetValueButton {hasDefaultValue} {onReset} />
		{/if}
	</div>
</div>
