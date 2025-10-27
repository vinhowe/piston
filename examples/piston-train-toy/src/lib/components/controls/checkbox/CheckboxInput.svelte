<script lang="ts">
	import KatexBlock from '../../KatexBlock.svelte';
	import Citations, { type CitationEntries as CitationsType } from '../Citations.svelte';
	import ResetValueButton from '../ResetValueButton.svelte';
	import CheckboxIcon from './CheckboxIcon.svelte';

	type $$Props = {
		label?: string;
		checked: boolean;
		citations?: CitationsType;
		id: string;
		class?: string;
		labelClass?: string;
		hasDefaultValue?: boolean;
		onReset?: () => void;
	};

	let {
		label,
		citations,
		checked = $bindable(),
		id,
		class: wrapperClass = '',
		labelClass = 'text-base',
		hasDefaultValue = false,
		onReset = undefined
	}: $$Props = $props();
</script>

<label
	for={id}
	class={`cursor-pointer select-none text-black grid grid-cols-[min-content_1fr] gap-x-1.25 shrink-0 ${wrapperClass}`.trim()}
>
	<div class="inline-flex items-center self-center justify-self-start">
		<input {id} type="checkbox" bind:checked class="sr-only peer" />
		<CheckboxIcon {checked} />
	</div>

	<div class="flex items-center justify-between gap-2 min-w-0">
		{#if label}
			<span class={labelClass}>
				<KatexBlock text={label} />
			</span>
		{/if}
		{#if onReset}
			<ResetValueButton {hasDefaultValue} {onReset} />
		{/if}
	</div>

	{#if citations}
		<div class="col-start-2 row-start-2 leading-none">
			<Citations {citations} />
		</div>
	{/if}
</label>
