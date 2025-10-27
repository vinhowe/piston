<script lang="ts">
	import type { ModelType } from '$lib/workspace/config';

	import { DATASET_CONFIG_METADATA } from '$lib/train/data';
	import { config, MODEL_TYPES } from '$lib/workspace/config.svelte';

	import Citations, { type CitationEntries as CitationsType } from './Citations.svelte';
	import SelectInput, { type SelectOption } from './select/SelectInput.svelte';

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
			description: metadata.description,
			type: metadata.type
		}));
	}

	const GROUP_LABELS = {
		natural: 'Natural Language',
		toy: 'Toy/Synthetic'
	};

	const options = $derived(getAvailableDatasets());
	const optionsGrouped = $derived.by(() => {
		const groups = options.reduce(
			(acc, opt) => {
				acc[opt.type] = acc[opt.type] || [];
				acc[opt.type].push(opt);
				return acc;
			},
			{} as Record<string, DatasetOption[]>
		);
		return Object.entries(groups).map(([type, options]) => ({
			label: GROUP_LABELS[type as keyof typeof GROUP_LABELS],
			options
		}));
	});

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

	const MODEL_DISPLAY_NAMES = {
		encoder: 'encoder',
		'encoder-decoder': 'encoder-decoder',
		decoder: 'decoder'
	};

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
	{@const supports = (
		'supportsModelTypes' in meta ? meta.supportsModelTypes : MODEL_TYPES
	).toSorted()}
	{@const currentModel = config.model.topology as ModelType}
	{@const isSupported = supports.includes(currentModel)}
	{@const autoModel = isSupported ? null : (supports[0] as ModelType)}
	<div class="leading-tight flex flex-col items-start">
		<span>
			<span class="mr-0.5">{name}</span>
			{#if autoModel}
				<span class="text-xs text-orange-800/50">
					(will switch to
					{MODEL_DISPLAY_NAMES[autoModel as keyof typeof MODEL_DISPLAY_NAMES]})
				</span>
			{/if}
		</span>
		{#if citations}
			<Citations {citations} />
		{/if}
		<div class="flex flex-wrap gap-1 mt-1">
			{#each supports as n (n)}
				<span
					class="border border-neutral-200/80 text-neutral-500 px-0.75 leading-4.5 text-2xs rounded-none"
					>{MODEL_DISPLAY_NAMES[n as keyof typeof MODEL_DISPLAY_NAMES]}</span
				>
			{/each}
		</div>
	</div>
{/snippet}

<div class="relative w-full">
	<SelectInput
		bind:value
		groups={optionsGrouped}
		{label}
		{id}
		class={wrapperClass}
		{hasDefaultValue}
		{onReset}
	>
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
