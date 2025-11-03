<script lang="ts">
	import type { ConfigParameter } from '$lib/train/data/toy/config';

	import { DATASET_CONFIG_METADATA } from '$lib/train/data';
	import { NATURAL_DATASET_META } from '$lib/train/data/natural';
	import { config, equalsConfigDefault, resetConfigToDefaults } from '$lib/workspace/config.svelte';
	import { getShowLowDiversityDatasetError } from '$lib/workspace/ui.svelte';

	import CheckboxInput from './checkbox/CheckboxInput.svelte';
	import ControlsNote from './ControlsNote.svelte';
	import DatasetSample from './DatasetSample.svelte';
	import RadioGroupInput from './radio/RadioGroupInput.svelte';
	import SelectDataset from './SelectDataset.svelte';
	import Slider from './Slider.svelte';
	let { datasetName }: { datasetName: typeof config.data.dataset } = $props();

	let datasetConfigMetadata = $derived(DATASET_CONFIG_METADATA[datasetName]);
	let parameters = $derived<Record<string, ConfigParameter> | undefined>(
		'parameters' in datasetConfigMetadata ? datasetConfigMetadata.parameters : undefined
	);
	let datasetConfig = $derived(
		datasetName in config.data.datasets
			? config.data.datasets[datasetName as keyof typeof config.data.datasets]
			: undefined
	);
	const showMaskRatio = $derived(config.model.topology === 'encoder');
	const isNatural = $derived(Object.keys(NATURAL_DATASET_META).includes(config.data.dataset));

	const showDivider = $derived(showMaskRatio);

	export function isSliderParameter(param: ConfigParameter): boolean {
		return param.type === 'number';
	}

	export function isCheckboxParameter(param: ConfigParameter): boolean {
		return param.type === 'boolean';
	}

	export function getSliderProps(param: ConfigParameter) {
		if (param.type !== 'number') return null;

		return {
			min: param.min || 0,
			max: param.max || 100,
			step: param.step || 1,
			default: param.default as number
		};
	}
</script>

<SelectDataset
	bind:value={config.data.dataset}
	id="dataset-control"
	hasDefaultValue={equalsConfigDefault('data.dataset')}
	onReset={() => resetConfigToDefaults('data.dataset')}
/>

<p class="-mt-1 mb-1 mx-0.5 text-sm py-1">{datasetConfigMetadata?.description}</p>

<DatasetSample />

{#if getShowLowDiversityDatasetError()}
	<div id="low-diversity-dataset-error" class="error-flash">
		<ControlsNote label="Low Diversity" type="error">
			<p>
				Not enough example diversity in the training dataset for a held-out validation set of size {config
					.training.validation.batchSize}. Consider changing dataset parameters or reducing the
				validation batch size.
			</p>
		</ControlsNote>
	</div>
{/if}

<div class="flex flex-col gap-1">
	{#key datasetName}
		{#if parameters && datasetConfig}
			<div class="flex flex-col gap-1">
				{#each Object.entries(parameters) as [paramKey, paramMeta] (paramKey)}
					{@const value = datasetConfig[paramKey as keyof typeof datasetConfig]}
					{#if isSliderParameter(paramMeta)}
						{@const sliderProps = getSliderProps(paramMeta)}
						{#if sliderProps}
							<Slider
								id={`dataset-${paramKey}`}
								label={paramMeta.name}
								bind:value={datasetConfig[paramKey as keyof typeof datasetConfig]}
								min={sliderProps.min}
								max={sliderProps.max}
								step={sliderProps.step}
								hasDefaultValue={value === paramMeta.default}
								onReset={() => {
									datasetConfig[paramKey as keyof typeof datasetConfig] =
										paramMeta.default as never;
								}}
							/>
						{/if}
					{:else if isCheckboxParameter(paramMeta)}
						<CheckboxInput
							id={`dataset-${paramKey}`}
							label={paramMeta.name}
							bind:checked={datasetConfig[paramKey as keyof typeof datasetConfig]}
							hasDefaultValue={value === paramMeta.default}
							onReset={() => {
								datasetConfig[paramKey as keyof typeof datasetConfig] = paramMeta.default as never;
							}}
						/>
					{/if}
				{/each}
			</div>
		{/if}
	{/key}
	{#if isNatural}
		<Slider
			id="dataset-context-size"
			label="Context Size"
			bind:value={config.data.natural.contextSize}
			min={8}
			max={1024}
			step={4}
			hasDefaultValue={equalsConfigDefault('data.natural.contextSize')}
			onReset={() => resetConfigToDefaults('data.natural.contextSize')}
		/>
		<RadioGroupInput
			id="dataset-natural-vocab-size"
			label="Tokenizer Vocabulary Size"
			name="natural-vocab-size"
			bind:value={
				() => String(config.data.natural.vocabSize),
				(v) =>
					(config.data.natural.vocabSize =
						v === 'char' ? 'char' : (parseInt(v) as typeof config.data.natural.vocabSize))
			}
			options={[
				{ value: 'char', label: 'Character-level' },
				{ value: '512', label: '512' },
				{ value: '1024', label: '1024' },
				{ value: '2048', label: '2048' },
				{ value: '4096', label: '4096' },
				{ value: '8192', label: '8192' },
				{ value: '16384', label: '16384' },
				{ value: '32768', label: '32768' },
				{ value: '65536', label: '65536' }
			]}
			hasDefaultValue={equalsConfigDefault('data.natural.vocabSize')}
			onReset={() => resetConfigToDefaults('data.natural.vocabSize')}
		/>
	{/if}
	{#if showDivider}
		<hr class="my-1 border-panel-border-base" />
	{/if}
	{#if showMaskRatio}
		<Slider
			id="dataset-mask-ratio"
			label="Mask Ratio"
			bind:value={config.data.maskRatio}
			min={0.01}
			max={1}
			step={0.01}
			hasDefaultValue={equalsConfigDefault('data.maskRatio')}
			onReset={() => resetConfigToDefaults('data.maskRatio')}
		/>
	{/if}
</div>
