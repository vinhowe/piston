<script lang="ts">
	import type { ModelType } from '$lib/workspace/config';

	import { DATASET_CONFIG_METADATA } from '$lib/train/data';
	import { config, MODEL_TYPES } from '$lib/workspace/config.svelte';

	import type { SelectOption } from './SelectInput.svelte';

	import Citations, { type CitationEntries as CitationsType } from '../Citations.svelte';
	import SelectInput from './SelectInput.svelte';

	type ModelOption = SelectOption<{
		title?: string;
		citations?: CitationsType;
		disabled?: boolean;
		reason?: string;
	}>;

	type $$Props = {
		value: string;
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

	const currentDatasetKey = $derived(config.data.dataset as keyof typeof DATASET_CONFIG_METADATA);
	const datasetMeta = $derived(DATASET_CONFIG_METADATA[currentDatasetKey]);
	const supportedModels = $derived(
		'supportsModelTypes' in datasetMeta
			? (datasetMeta.supportsModelTypes as readonly ModelType[])
			: MODEL_TYPES
	);

	function optionDisabled(opt: ModelOption): boolean {
		return !supportedModels.includes(opt.value as ModelType) || Boolean(opt.disabled);
	}

	const modelTypeOptions: ModelOption[] = $derived([
		{
			value: 'decoder',
			title: 'Decoder-only',
			citations:
				config.model.family === 'transformer'
					? {
							entries: [
								{
									name: 'Radford et al., 2019',
									url: 'https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf'
								}
							],
							extra: '; GPT-series'
						}
					: {
							entries: [
								{
									name: 'Mikolov et al., 2010',
									url: 'https://www.isca-archive.org/interspeech_2010/mikolov10_interspeech.html'
								}
							],
							extra: '; RNN LM'
						}
		},
		{
			value: 'encoder-decoder',
			title: 'Encoder-Decoder',
			citations:
				config.model.family === 'transformer'
					? {
							entries: [
								{
									name: 'Vaswani et al., 2017',
									url: 'https://arxiv.org/abs/1706.03762'
								}
							],
							extra: '; Transformer'
						}
					: {
							entries: [
								{
									name: 'Sutskever et al.',
									url: 'https://papers.nips.cc/paper_files/paper/2014/hash/5a18e133cbf9f257297f410bb7eca942-Abstract.html'
								},
								{
									name: 'Cho et al., 2014',
									url: 'https://aclanthology.org/D14-1179/'
								}
							],
							extra: '; seq2seq'
						}
		},
		{
			value: 'encoder',
			title: 'Encoder-only',
			citations:
				config.model.family === 'transformer'
					? {
							entries: [
								{ name: 'Devlin et al., 2019; BERT', url: 'https://aclanthology.org/N19-1423/' }
							]
						}
					: {
							entries: [
								{
									name: 'Schuster & Paliwal, 1997; BiRNN',
									url: 'https://ieeexplore.ieee.org/document/650093'
								}
							]
						}
		}
	]);

	const processedOptions = $derived(
		modelTypeOptions.map((o) => ({ ...o, disabled: optionDisabled(o) }))
	);

	const selectedOption = $derived(processedOptions.find((o) => o.value === value));
</script>

{#snippet optionView(_opt: unknown, _selected: boolean, _index: number)}
	{@const opt = _opt as ModelOption}
	{@const label = opt.title}
	{@const citations = opt.citations}
	<div class="leading-tight">
		<div>{label}</div>
		{#if citations}
			<Citations {citations} />
		{/if}
		{#if opt.disabled}
			{@const datasetName = DATASET_CONFIG_METADATA[config.data.dataset].name}
			<div class="text-xs font-medium mt-0.5 text-orange-900">Not supported by {datasetName}</div>
		{/if}
	</div>
{/snippet}

<div class="relative w-full">
	<SelectInput
		bind:value
		{label}
		{id}
		class={wrapperClass}
		{hasDefaultValue}
		{onReset}
		options={processedOptions}
	>
		{#snippet option(_opt, _selected, _i)}
			{@render optionView(_opt, _selected, _i)}
		{/snippet}
		{#snippet trigger(_selected)}
			{#if _selected}
				{@const opt = _selected as ModelOption}
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
