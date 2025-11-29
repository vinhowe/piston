<script lang="ts">
	import {
		SelectInput,
		type SelectOption,
		Citations,
		type CitationEntries as CitationsType
	} from 'example-common';

	type CitationOption = SelectOption<{
		title?: string;
		citations?: CitationsType;
	}>;

	type $$Props = {
		value: string | number;
		options: CitationOption[];
		label?: string;
		id: string;
		class?: string;
		hasDefaultValue?: boolean;
		onReset?: () => void;
	};

	let {
		value = $bindable(),
		options,
		label,
		id,
		class: wrapperClass = '',
		hasDefaultValue = false,
		onReset
	}: $$Props = $props();

	const selectedOption = $derived(options.find((o) => o.value === value));
</script>

{#snippet citationView(_opt: CitationOption)}
	{@const label = _opt.title}
	{@const citations = _opt.citations}
	<div class="leading-tight">
		<div>{label}</div>
		{#if citations}
			<Citations {citations} />
		{/if}
	</div>
{/snippet}

<div class="relative w-full">
	<SelectInput bind:value {options} {label} {id} class={wrapperClass} {hasDefaultValue} {onReset}>
		{#snippet option(_opt, _selected)}
			{@const opt = _opt as CitationOption}
			{@render citationView(opt)}
		{/snippet}
		{#snippet trigger(_selected)}
			{#if _selected}
				{@const opt = _selected as CitationOption}
				{opt.title}
			{:else}
				Select...
			{/if}
		{/snippet}
	</SelectInput>
	{#if selectedOption?.citations}
		<div class="ml-px leading-none"><Citations citations={selectedOption.citations} /></div>
	{/if}
</div>
