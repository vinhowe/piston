<script lang="ts">
	import { DATASET_CONFIG_METADATA } from '$lib/train/data';

	import {
		SelectInput,
		type SelectOption,
		Citations,
		type CitationEntries as CitationsType
	} from 'example-common';

	type DatasetName = keyof typeof DATASET_CONFIG_METADATA;

	type DatasetOption = SelectOption<{
		text?: string;
		description?: string;
		citations?: CitationsType;
	}>;

	function getAvailableDatasets() {
		return Object.entries(DATASET_CONFIG_METADATA).map(([key, metadata]) => ({
			value: key,
			text: metadata.name,
			description: metadata.description
		}));
	}

	const options = $derived(getAvailableDatasets());

	type $$Props = {
		value: DatasetName;
		label?: string;
		id: string;
		class?: string;
		hasDefaultValue?: boolean;
		onReset?: () => void;
	};

	let {
		value = $bindable(),
		label,
		id,
		class: wrapperClass = '',
		hasDefaultValue = false,
		onReset
	}: $$Props = $props();

	function getMeta(name: DatasetName) {
		return DATASET_CONFIG_METADATA[name];
	}

	const selectedOption = $derived(options.find((o) => o.value === value));
	const selectedMeta = $derived(getMeta(selectedOption?.value as DatasetName));
</script>

{#snippet nameAndBadges(_opt: DatasetOption)}
	{@const opt = _opt as DatasetOption}
	{@const meta = getMeta(opt.value as DatasetName)}
	{@const name = opt.text ?? meta.name}
	{@const citations =
		'citations' in meta
			? ((meta as Record<string, unknown>).citations as CitationsType)
			: undefined}
	<div class="leading-tight flex flex-col items-start">
		<span class="mr-0.5">{name}</span>
		{#if citations}
			<Citations {citations} />
		{/if}
	</div>
{/snippet}

<div class="relative w-full">
	<SelectInput bind:value {options} {label} {id} class={wrapperClass} {hasDefaultValue} {onReset}>
		{#snippet option(_opt, _selected)}
			{@const opt = _opt as DatasetOption}
			{@render nameAndBadges(opt)}
		{/snippet}
		{#snippet trigger(_selected)}
			{#if _selected}
				{@const opt = _selected as DatasetOption}
				{opt.text ?? opt.value}
			{:else}
				Select...
			{/if}
		{/snippet}
	</SelectInput>
	{#if 'citations' in selectedMeta}
		<div class="ml-px leading-none">
			<Citations citations={(selectedMeta as { citations: CitationsType }).citations} />
		</div>
	{/if}
</div>
