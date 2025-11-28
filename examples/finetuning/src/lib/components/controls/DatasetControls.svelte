<script lang="ts">
	import { DATASET_CONFIG_METADATA } from '$lib/train/data';
	import { NATURAL_DATASET_META } from '$lib/train/data/natural';
	import { config, equalsConfigDefault, resetConfigToDefaults } from '$lib/workspace/config.svelte';
	import { getShowLowDiversityDatasetError } from '$lib/workspace/ui.svelte';

	import { ControlsNote } from 'example-common';
	import DatasetSample from './DatasetSample.svelte';
	import SelectDataset from './SelectDataset.svelte';
	import { Slider, RadioGroupInput } from 'example-common';
	let { datasetName }: { datasetName: typeof config.data.dataset } = $props();

	let datasetConfigMetadata = $derived(DATASET_CONFIG_METADATA[datasetName]);
	const isNatural = $derived(Object.keys(NATURAL_DATASET_META).includes(config.data.dataset));
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
