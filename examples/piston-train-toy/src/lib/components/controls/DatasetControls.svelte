<script lang="ts">
	import type { ConfigParameter } from '$lib/train/data/toy/config';

	import { DATASET_CONFIG_METADATA } from '$lib/train/data';
	import { config, equalsConfigDefault, resetConfigToDefaults } from '$lib/workspace/config.svelte';

	import CheckboxInput from './checkbox/CheckboxInput.svelte';
	import DatasetSample from './DatasetSample.svelte';
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
