<script lang="ts">
	export const ssr = false;

	import {
		config,
		equalsConfigDefault,
		getCurrentDataset,
		getHiddenSize,
		getMlpIntermediateSize,
		resetConfigToDefaults,
		validateConfig
	} from '$lib/workspace/config.svelte';
	import { controlSectionsOpen, toggleControlSection } from '$lib/workspace/ui.svelte';
	import { getParameterCount, triggerModelInspection } from '$lib/workspace/workers.svelte';
	import { untrack } from 'svelte';

	import ActivationPicker from './ActivationPicker.svelte';
	import BorderedGroup from './BorderedGroup.svelte';
	import CheckboxInput from './checkbox/CheckboxInput.svelte';
	import CollapsibleSection from './CollapsibleSection.svelte';
	import ControlsStatistic from './ControlsStatistic.svelte';
	import DatasetControls from './DatasetControls.svelte';
	import LRSchedulePicker from './LRSchedulePicker.svelte';
	import NumberInput from './NumberInput.svelte';
	import SelectInput from './select/SelectInput.svelte';
	import SelectModelTopology from './select/SelectModelTopology.svelte';
	import SelectWithCitations from './select/SelectWithCitations.svelte';
	import Slider from './Slider.svelte';
	import TextInput from './TextInput.svelte';
	import ToggleGroup from './ToggleGroup.svelte';

	const sectionClass = 'p-1 space-y-1 flex flex-col';
	// Put a little bit of padding at the bottom of collapsible sections to demarcate the end of the section
	const collapsibleSectionClass = 'p-1 space-y-1 flex flex-col';

	// Model comparison data for parameter count display
	const attentionModelComparisons = {
		decoder: { name: 'GPT-2 small', params: 124_000_000 },
		encoder: { name: 'BERT base', params: 110_000_000 },
		'encoder-decoder': { name: 'T5 Small', params: 60_000_000 }
	};

	// Derived reactive variable for parameter comparison
	const parameterComparison = $derived.by(() => {
		const paramCount = getParameterCount();
		const modelType = config.model.topology;

		if (!paramCount || !(modelType in attentionModelComparisons)) return null;

		const comparison =
			attentionModelComparisons[modelType as keyof typeof attentionModelComparisons];
		const nonBreakingName = comparison.name.replace(' ', '\xa0').replace('-', '\u2011');
		const percentage = Math.round((paramCount / comparison.params) * 10000) / 100;
		return `(${percentage}% of ${nonBreakingName} @ ${comparison.params / 1_000_000}M)`;
	});

	const currentDataset = $derived(getCurrentDataset());

	$effect(() => {
		validateConfig();
	});

	// Reactive parameter counting; trigger when model-relevant config changes
	$effect(() => {
		// Add dependency on both model and data config (data config influences vocab size)
		JSON.stringify(config.model);
		JSON.stringify(config.data);

		// Trigger parameter count when config changes, but triggerParameterCount itself needs to be untracked
		untrack(() => triggerModelInspection());
	});
</script>

{#snippet parameterComparisonFunFact()}
	{parameterComparison}
{/snippet}

{#snippet projectionBlock(configName: 'attention' | 'mlp' | 'lmHead', displayName: string)}
	<ToggleGroup
		id={`projections-control-${configName}`}
		title={displayName}
		showEnableToggle={true}
		bind:enabled={config.model.transformer.initialization.projections[configName].present}
		hasDefaultValue={equalsConfigDefault(
			`model.transformer.initialization.projections.${configName}.present`
		)}
		onReset={() =>
			resetConfigToDefaults(`model.transformer.initialization.projections.${configName}.present`)}
	>
		<SelectInput
			id={`projections-strategy-${configName}`}
			bind:value={config.model.transformer.initialization.projections[configName].strategy}
			options={[
				{ value: 'layer-scaled', text: 'Layer-Scaled' },
				{ value: 'zero', text: 'Zero' }
			]}
			hasDefaultValue={equalsConfigDefault(
				`model.transformer.initialization.projections.${configName}.strategy`
			)}
			onReset={() =>
				resetConfigToDefaults(
					`model.transformer.initialization.projections.${configName}.strategy`
				)}
		/>
	</ToggleGroup>
{/snippet}

<div>
	<!-- Task -->
	<CollapsibleSection
		title="Task"
		isOpen={controlSectionsOpen.current.task}
		ontoggle={() => toggleControlSection('task')}
		contentClass={collapsibleSectionClass}
	>
		<DatasetControls datasetName={config.data.dataset} />
	</CollapsibleSection>

	<!-- Model -->
	<CollapsibleSection
		title="Model Architecture"
		isOpen={controlSectionsOpen.current.model}
		ontoggle={() => toggleControlSection('model')}
		contentClass={collapsibleSectionClass}
	>
		<!-- Parameter count display -->
		<div class="-space-y-px">
			<ControlsStatistic
				label="Trainable Parameters"
				funFact={parameterComparison ? parameterComparisonFunFact : undefined}
			>
				{#if getParameterCount() !== null}
					{getParameterCount()?.toLocaleString()}
				{:else}
					...,...
				{/if}
			</ControlsStatistic>
			<ControlsStatistic label="Embedding Size ($d_&lbrace;\text&lbrace;model&rbrace;&rbrace;$)">
				{getHiddenSize()}
			</ControlsStatistic>
			<ControlsStatistic label="Token Vocabulary Size">
				{currentDataset.vocabSize}
			</ControlsStatistic>
			{#if config.model.topology === 'encoder-decoder'}
				{@const blockSize = currentDataset.blockSize as { source: number; target: number }}
				<ControlsStatistic label="Max Input Length">
					{blockSize.source}
				</ControlsStatistic>
				<ControlsStatistic label="Max Output Length">
					{blockSize.target}
				</ControlsStatistic>
			{:else}
				<ControlsStatistic label="Max Context Length">
					{currentDataset.blockSize}
				</ControlsStatistic>
			{/if}
		</div>

		<SelectModelTopology
			id="model-topology"
			bind:value={config.model.topology}
			label="Topology"
			hasDefaultValue={equalsConfigDefault('model.topology')}
			onReset={() => resetConfigToDefaults('model.topology')}
		/>

		{#if config.model.topology !== 'encoder-decoder'}
			<Slider
				id="model-layers"
				label={config.model.topology === 'decoder'
					? 'Decoder Layers ($N_d$)'
					: 'Encoder Layers ($N_e$)'}
				bind:value={config.model.layers}
				min={1}
				max={32}
				step={1}
				base={2}
				hasDefaultValue={equalsConfigDefault('model.layers')}
				onReset={() => resetConfigToDefaults('model.layers')}
			/>
		{:else}
			<Slider
				id="model-encoder-layers"
				label="Encoder Layers ($N_e$)"
				bind:value={config.model.encoderDecoder.encoderLayers}
				min={1}
				max={32}
				step={1}
				base={2}
				hasDefaultValue={equalsConfigDefault('model.encoderDecoder.encoderLayers')}
				onReset={() => resetConfigToDefaults('model.encoderDecoder.encoderLayers')}
			/>
			<Slider
				id="model-decoder-layers"
				label="Decoder Layers ($N_d$)"
				bind:value={config.model.encoderDecoder.decoderLayers}
				min={1}
				max={32}
				step={1}
				base={2}
				hasDefaultValue={equalsConfigDefault('model.encoderDecoder.decoderLayers')}
				onReset={() => resetConfigToDefaults('model.encoderDecoder.decoderLayers')}
			/>
		{/if}

		<Slider
			id="model-head-dim"
			label="Head Dimension ($d_&lbrace;\text&lbrace;model&rbrace;&rbrace;/h$)"
			bind:value={config.model.transformer.headDim}
			min={8}
			max={64}
			step={8}
			base={2}
			hasDefaultValue={equalsConfigDefault('model.transformer.headDim')}
			onReset={() => resetConfigToDefaults('model.transformer.headDim')}
		/>

		<ToggleGroup
			id="model-round-vocab-size-to-nearest-multiple-group"
			title="Round Token Vocabulary Size to Nearest Multiple"
			citations={{
				entries: [
					{ name: 'Karpathy, 2023', url: 'https://x.com/karpathy/status/1621578354024677377' },
					{
						name: 'NVIDIA Docs',
						url: 'http://web.archive.org/web/20250912215737/https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html#requirements-tc__table_usc_swx_lmb'
					}
				]
			}}
			showEnableToggle={true}
			bind:enabled={config.model.roundVocabSizeToNearestMultiple.present}
			hasDefaultValue={equalsConfigDefault('model.roundVocabSizeToNearestMultiple.present')}
			onReset={() => resetConfigToDefaults('model.roundVocabSizeToNearestMultiple.present')}
		>
			<Slider
				id="model-round-vocab-size-to-nearest-multiple-value"
				bind:value={config.model.roundVocabSizeToNearestMultiple.value}
				min={2}
				max={1024}
				step={8}
				base={2}
				hasDefaultValue={equalsConfigDefault('model.roundVocabSizeToNearestMultiple.value')}
				onReset={() => resetConfigToDefaults('model.roundVocabSizeToNearestMultiple.value')}
			/>
		</ToggleGroup>

		<!-- Attention -->
		<ToggleGroup
			id="model-attention-control"
			title="Multi-Head Attention (MHA)"
			showEnableToggle={true}
			citations={{
				entries: [{ name: 'Vaswani et al., 2017', url: 'https://arxiv.org/abs/1706.03762' }]
			}}
			contentClass={sectionClass}
			bind:enabled={config.model.transformer.attention.present}
			hasDefaultValue={equalsConfigDefault('model.transformer.attention.present')}
			onReset={() => resetConfigToDefaults('model.transformer.attention.present')}
		>
			<Slider
				id="model-attention-n-key-value-heads"
				label="Number of Key-Value Heads ($h_&lbrace;\text&lbrace;kv&rbrace;&rbrace;$)"
				bind:value={config.model.transformer.attention.nKeyValueHeads}
				min={1}
				max={32}
				step={1}
				base={2}
				hasDefaultValue={equalsConfigDefault('model.transformer.attention.nKeyValueHeads')}
				onReset={() => resetConfigToDefaults('model.transformer.attention.nKeyValueHeads')}
			/>
			<ToggleGroup
				id="model-attention-grouped-query-attention-group"
				title="Grouped-Query Attention (GQA)"
				citations={{
					entries: [
						{ name: 'Ainslie et al., 2023', url: 'https://aclanthology.org/2023.emnlp-main.298/' }
					]
				}}
				showEnableToggle={true}
				bind:enabled={config.model.transformer.attention.groupedQueryAttention.present}
				contentClass={sectionClass}
				hasDefaultValue={equalsConfigDefault(
					'model.transformer.attention.groupedQueryAttention.present'
				)}
				onReset={() =>
					resetConfigToDefaults('model.transformer.attention.groupedQueryAttention.present')}
			>
				<Slider
					id="model-attention-grouped-query-attention-query-heads-per-key-value-head"
					label="Number of Query Heads per Key-Value Head ($h_&lbrace;\text&lbrace;q&rbrace;&rbrace;/h_&lbrace;\text&lbrace;kv&rbrace;&rbrace;$)"
					bind:value={
						config.model.transformer.attention.groupedQueryAttention.queryHeadsPerKeyValueHead
					}
					min={2}
					max={32}
					step={1}
					base={2}
					hasDefaultValue={equalsConfigDefault(
						'model.transformer.attention.groupedQueryAttention.queryHeadsPerKeyValueHead'
					)}
					onReset={() =>
						resetConfigToDefaults(
							'model.transformer.attention.groupedQueryAttention.queryHeadsPerKeyValueHead'
						)}
				/>
			</ToggleGroup>
		</ToggleGroup>

		<!-- MLP -->
		<ToggleGroup
			id="mlp-control"
			title="Feed-Forward Network (MLP)"
			showEnableToggle={true}
			contentClass={sectionClass}
			bind:enabled={config.model.transformer.mlp.present}
			hasDefaultValue={equalsConfigDefault('model.transformer.mlp.present')}
			onReset={() => resetConfigToDefaults('model.transformer.mlp.present')}
		>
			<ControlsStatistic label="MLP hidden dimension ($d_&lbrace;\text&lbrace;ff&rbrace;&rbrace;$)">
				{getMlpIntermediateSize()}
			</ControlsStatistic>
			<ActivationPicker
				id="model-mlp-activation"
				bind:value={config.model.transformer.mlp.activation}
				hasDefaultValue={equalsConfigDefault('model.transformer.mlp.activation')}
				onReset={() => resetConfigToDefaults('model.transformer.mlp.activation')}
			/>
			<Slider
				id="model-mlp-hidden-expansion-factor"
				label="Hidden Expansion Factor ($d_&lbrace;\text&lbrace;ff&rbrace;&rbrace;/d_&lbrace;\text&lbrace;model&rbrace;&rbrace;$)"
				bind:value={config.model.transformer.mlp.hiddenExpansionFactor}
				min={1}
				max={16}
				base={2}
				step={1}
				hasDefaultValue={equalsConfigDefault('model.transformer.mlp.hiddenExpansionFactor')}
				onReset={() => resetConfigToDefaults('model.transformer.mlp.hiddenExpansionFactor')}
			/>
		</ToggleGroup>

		<ToggleGroup
			id="model-layer-normalization-control"
			title="Positional Encoding"
			showEnableToggle={true}
			bind:enabled={config.model.transformer.positionalEncoding.present}
			hasDefaultValue={equalsConfigDefault('model.transformer.positionalEncoding.present')}
			onReset={() => resetConfigToDefaults('model.transformer.positionalEncoding.present')}
		>
			<SelectWithCitations
				id="model-layer-normalization-type"
				bind:value={config.model.transformer.positionalEncoding.type}
				options={[
					{
						value: 'sinusoidal',
						title: 'Sinusoidal',
						citations: {
							entries: [{ name: 'Vaswani et al., 2017', url: 'https://arxiv.org/abs/1706.03762' }],
							extra: '; Transformer'
						}
					},
					{
						value: 'learned',
						title: 'Learned',
						citations: {
							entries: [
								{
									name: 'Radford et al., 2018',
									url: 'https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf'
								}
							],
							extra: '; GPT-series'
						}
					},
					{
						value: 'rope',
						title: 'Rotary',
						citations: {
							entries: [
								{
									name: 'Su et al., 2021',
									url: 'https://www.sciencedirect.com/science/article/pii/S0925231223011864'
								}
							],
							extra: '; Llama'
						}
					},
					{
						value: 'alibi',
						title: 'Alibi',
						citations: {
							entries: [
								{ name: 'Press et al., 2021', url: 'https://openreview.net/forum?id=R8sQPpGCv0' }
							],
							extra: '; BLOOM, MPT'
						}
					}
				]}
				hasDefaultValue={equalsConfigDefault('model.transformer.positionalEncoding.type')}
				onReset={() => resetConfigToDefaults('model.transformer.positionalEncoding.type')}
			/>
			{#if config.model.transformer.positionalEncoding.type === 'rope'}
				<Slider
					id="model-positional-encoding-rope-base"
					label="Base"
					bind:value={config.model.transformer.positionalEncoding.rope.base}
					min={1}
					max={10000}
					step={1}
					useLog
					hasDefaultValue={equalsConfigDefault('model.transformer.positionalEncoding.rope.base')}
					onReset={() => resetConfigToDefaults('model.transformer.positionalEncoding.rope.base')}
				/>
			{/if}
			{#if config.model.transformer.positionalEncoding.type === 'alibi'}
				<Slider
					id="model-positional-encoding-alibi-max-bias"
					label="Max Bias"
					bind:value={config.model.transformer.positionalEncoding.alibi.maxBias}
					min={1}
					max={100}
					step={1}
					hasDefaultValue={equalsConfigDefault(
						'model.transformer.positionalEncoding.alibi.maxBias'
					)}
					onReset={() =>
						resetConfigToDefaults('model.transformer.positionalEncoding.alibi.maxBias')}
				/>
			{/if}
		</ToggleGroup>

		<!-- Normalization -->
		<BorderedGroup title="Normalization" contentClass={sectionClass}>
			<ToggleGroup
				id="model-layer-normalization-control"
				title="Layer Normalization"
				showEnableToggle={true}
				bind:enabled={config.model.layerNormalization.transformer.present}
				contentClass={sectionClass}
				hasDefaultValue={equalsConfigDefault('model.layerNormalization.transformer.present')}
				onReset={() => resetConfigToDefaults('model.layerNormalization.transformer.present')}
			>
				<div id="model-layer-normalization-control">
					<SelectWithCitations
						id="model-layer-normalization-type"
						bind:value={config.model.layerNormalization.type}
						options={[
							{
								value: 'layernorm',
								title: 'LayerNorm',
								citations: {
									entries: [
										{
											name: 'Ba et al., 2016',
											url: 'https://www.cs.utoronto.ca/~hinton/absps/LayerNormalization.pdf'
										}
									],
									extra: '; Transformer, GPT-series'
								}
							},
							{
								value: 'rmsnorm',
								title: 'RMSNorm',
								citations: {
									entries: [
										{
											name: 'Zhang & Sennrich, 2019',
											url: 'https://proceedings.neurips.cc/paper_files/paper/2019/file/1e8a19426224ca89e83cef47f1e7f53b-Paper.pdf'
										}
									],
									extra: '; Llama'
								}
							}
						]}
						hasDefaultValue={equalsConfigDefault('model.layerNormalization.type')}
						onReset={() => resetConfigToDefaults('model.layerNormalization.type')}
					/>
					<SelectWithCitations
						id="model-layer-normalization-position"
						label="Position"
						bind:value={config.model.layerNormalization.transformer.position}
						options={[
							{
								value: 'post',
								title: 'Post-Norm',
								citations: { extra: 'Transformer, GPT' }
							},
							{
								value: 'pre',
								title: 'Pre-Norm',
								citations: {
									extra: 'GPT-2, most recent models'
								}
							}
						]}
						hasDefaultValue={equalsConfigDefault('model.layerNormalization.transformer.position')}
						onReset={() => resetConfigToDefaults('model.layerNormalization.transformer.position')}
					/>
					<Slider
						id="model-layer-normalization-epsilon"
						label="Epsilon"
						bind:value={config.model.layerNormalization.eps}
						min={1e-5}
						max={1e-1}
						step={1e-5}
						useLog
						hasDefaultValue={equalsConfigDefault('model.layerNormalization.eps')}
						onReset={() => resetConfigToDefaults('model.layerNormalization.eps')}
					/>
				</div>
			</ToggleGroup>
		</BorderedGroup>
		<ToggleGroup
			id="initialization-control"
			title="Initialization"
			showEnableToggle={true}
			bind:enabled={config.model.transformer.initialization.present}
			contentClass={sectionClass}
			hasDefaultValue={equalsConfigDefault(`model.transformer.initialization.present`)}
			onReset={() => resetConfigToDefaults(`model.transformer.initialization.present`)}
		>
			<Slider
				id="initialization-std"
				label="Standard Deviation ($&sigma;$)"
				bind:value={config.model.transformer.initialization.std}
				min={0}
				max={100}
				useLog
				step={0.0001}
				hasDefaultValue={equalsConfigDefault('model.transformer.initialization.std')}
				onReset={() => resetConfigToDefaults('model.transformer.initialization.std')}
			/>
			<BorderedGroup contentClass={sectionClass} title="Initialize Projections Separately">
				{@render projectionBlock('attention', 'Attention')}
				{@render projectionBlock('mlp', 'MLP')}
				{@render projectionBlock('lmHead', 'LM Head')}
			</BorderedGroup>
		</ToggleGroup>
	</CollapsibleSection>

	<!-- Training -->
	<CollapsibleSection
		title="Training"
		isOpen={controlSectionsOpen.current.training}
		contentClass={collapsibleSectionClass}
		ontoggle={() => toggleControlSection('training')}
	>
		<Slider
			id="training-log-steps"
			label="Log Metrics Every $n$ Steps"
			bind:value={config.training.logSteps}
			min={1}
			max={100}
			step={1}
			hasDefaultValue={equalsConfigDefault('training.logSteps')}
			onReset={() => resetConfigToDefaults('training.logSteps')}
		/>

		<Slider
			id="training-batch-size"
			label="Mini-Batch Size"
			bind:value={config.training.batchSize}
			min={1}
			max={1024}
			step={1}
			base={2}
			hasDefaultValue={equalsConfigDefault('training.batchSize')}
			onReset={() => resetConfigToDefaults('training.batchSize')}
		/>

		<CheckboxInput
			label="Enable Activation Visualization"
			bind:checked={config.training.enableVisualization}
			hasDefaultValue={equalsConfigDefault('training.enableVisualization')}
			onReset={() => resetConfigToDefaults('training.enableVisualization')}
		/>

		<ToggleGroup
			id="training-grad-norm-group"
			title="Track Gradient Norm"
			showEnableToggle={true}
			bind:enabled={config.training.gradNorm.track}
			contentClass={sectionClass}
			hasDefaultValue={equalsConfigDefault('training.gradNorm.track')}
			onReset={() => resetConfigToDefaults('training.gradNorm.track')}
		>
			<CheckboxInput
				id="training-grad-norm-error-if-nonfinite"
				label="Raise Error if Nonfinite"
				bind:checked={config.training.gradNorm.errorIfNonfinite}
				hasDefaultValue={equalsConfigDefault('training.gradNorm.errorIfNonfinite')}
				onReset={() => resetConfigToDefaults('training.gradNorm.errorIfNonfinite')}
			/>

			<ToggleGroup
				id="training-clip-grad-norm-group"
				title="Clip Gradient Norm"
				showEnableToggle={true}
				bind:enabled={config.training.clipGradNorm.present}
				contentClass={sectionClass}
				hasDefaultValue={equalsConfigDefault('training.clipGradNorm.present')}
				onReset={() => resetConfigToDefaults('training.clipGradNorm.present')}
			>
				<Slider
					id="training-clip-grad-max-norm-value"
					label="Max Norm"
					bind:value={config.training.clipGradNorm.value}
					min={0}
					max={10}
					step={0.01}
					hasDefaultValue={equalsConfigDefault('training.clipGradNorm.value')}
					onReset={() => resetConfigToDefaults('training.clipGradNorm.value')}
				/>
			</ToggleGroup>
		</ToggleGroup>

		<ToggleGroup
			id="validation-control"
			title="Validation"
			showEnableToggle={true}
			bind:enabled={config.training.validation.present}
			contentClass={sectionClass}
			hasDefaultValue={equalsConfigDefault('training.validation.present')}
			onReset={() => resetConfigToDefaults('training.validation.present')}
		>
			<Slider
				id="training-validation-val-steps"
				label="Validate Every $k$ Steps"
				bind:value={config.training.validation.valSteps}
				min={1}
				max={100}
				step={1}
				hasDefaultValue={equalsConfigDefault('training.validation.valSteps')}
				onReset={() => resetConfigToDefaults('training.validation.valSteps')}
			/>
			<Slider
				label="Number of Examples Held-out for Validation"
				id="training-validation-batch-size"
				bind:value={config.training.validation.batchSize}
				min={1}
				max={1024}
				step={1}
				base={2}
				hasDefaultValue={equalsConfigDefault('training.validation.batchSize')}
				onReset={() => resetConfigToDefaults('training.validation.batchSize')}
			/>
			<ToggleGroup
				id="validation-completions-control"
				title="Generation"
				contentClass={sectionClass}
				showEnableToggle={true}
				bind:enabled={config.training.validation.completions.present}
				hasDefaultValue={equalsConfigDefault('training.validation.completions.present')}
				onReset={() => resetConfigToDefaults('training.validation.completions.present')}
			>
				<SelectInput
					id="validation-completions-amount"
					bind:value={config.training.validation.completions.amount}
					options={[
						{ value: 'all', text: 'All Validation Examples' },
						{ value: 'subset', text: 'Subset of Validation Examples' }
					]}
					hasDefaultValue={equalsConfigDefault('training.validation.completions.amount')}
					onReset={() => resetConfigToDefaults('training.validation.completions.amount')}
				/>
				{#if config.training.validation.completions.amount === 'subset'}
					<Slider
						id="validation-completions-subset-size"
						bind:value={config.training.validation.completions.subsetSize}
						min={1}
						max={config.training.validation.batchSize}
						step={1}
						base={2}
						hasDefaultValue={equalsConfigDefault('training.validation.completions.subsetSize')}
						onReset={() => resetConfigToDefaults('training.validation.completions.subsetSize')}
					/>
				{/if}
				<BorderedGroup title="Sampling" contentClass={sectionClass}>
					<Slider
						id="validation-completions-temperature"
						label="Temperature"
						bind:value={config.training.validation.temperature}
						min={0}
						max={1.2}
						step={0.01}
						hasDefaultValue={equalsConfigDefault('training.validation.temperature')}
						onReset={() => resetConfigToDefaults('training.validation.temperature')}
					/>
				</BorderedGroup>
			</ToggleGroup>
		</ToggleGroup>

		<ToggleGroup
			id="training-limit-training-steps-group"
			title="Limit Training Steps"
			showEnableToggle={true}
			bind:enabled={config.training.limitTraining.present}
			contentClass={sectionClass}
			hasDefaultValue={equalsConfigDefault('training.limitTraining.present')}
			onReset={() => resetConfigToDefaults('training.limitTraining.present')}
		>
			<Slider
				id="training-limit-training-steps-value"
				bind:value={config.training.limitTraining.steps}
				min={1}
				max={50_000}
				step={1}
				hasDefaultValue={equalsConfigDefault('training.limitTraining.steps')}
				onReset={() => resetConfigToDefaults('training.limitTraining.steps')}
			/>
		</ToggleGroup>

		<BorderedGroup title="Regularization" contentClass={sectionClass}>
			<!-- Dropout -->
			<ToggleGroup
				id="dropout-control"
				title="Dropout"
				citations={{
					entries: [
						{
							name: 'Srivastava et al., 2014',
							url: 'https://jmlr.org/papers/v15/srivastava14a.html'
						}
					]
				}}
				showEnableToggle={true}
				bind:enabled={config.training.dropout.present}
				contentClass={sectionClass}
				hasDefaultValue={equalsConfigDefault('training.dropout.present')}
				onReset={() => resetConfigToDefaults('training.dropout.present')}
			>
				<Slider
					id="training-dropout-embedding"
					label="Embedding Dropout Rate"
					bind:value={config.training.dropout.embedding}
					min={0}
					max={1}
					step={0.01}
					hasDefaultValue={equalsConfigDefault('training.dropout.embedding')}
					onReset={() => resetConfigToDefaults('training.dropout.embedding')}
				/>
				<Slider
					id="training-dropout-attention"
					label="Attention Dropout Rate"
					bind:value={config.training.dropout.transformer.attention}
					min={0}
					max={1}
					step={0.01}
					hasDefaultValue={equalsConfigDefault('training.dropout.transformer.attention')}
					onReset={() => resetConfigToDefaults('training.dropout.transformer.attention')}
				/>
				<Slider
					id="training-dropout-residual"
					label="Residual Dropout Rate"
					bind:value={config.training.dropout.transformer.residual}
					min={0}
					max={1}
					step={0.01}
					hasDefaultValue={equalsConfigDefault('training.dropout.transformer.residual')}
					onReset={() => resetConfigToDefaults('training.dropout.transformer.residual')}
				/>
			</ToggleGroup>
		</BorderedGroup>

		<ToggleGroup
			id="random-seed-control"
			title="Random Seed"
			showEnableToggle={true}
			bind:enabled={config.training.randomSeed.present}
			hasDefaultValue={equalsConfigDefault('training.randomSeed.present')}
			onReset={() => resetConfigToDefaults('training.randomSeed.present')}
		>
			<TextInput
				id="random-seed-input"
				bind:value={config.training.randomSeed.value}
				hasDefaultValue={equalsConfigDefault('training.randomSeed.value')}
				onReset={() => resetConfigToDefaults('training.randomSeed.value')}
			/>
		</ToggleGroup>

		<ToggleGroup
			id="training-vram-limit-group"
			title="GPU Memory Limit"
			showEnableToggle={true}
			bind:enabled={config.training.vramLimitMb.present}
			hasDefaultValue={equalsConfigDefault('training.vramLimitMb.present')}
			onReset={() => resetConfigToDefaults('training.vramLimitMb.present')}
		>
			<Slider
				id="training-vram-limit-value"
				bind:value={config.training.vramLimitMb.value}
				unit="MB"
				min={1}
				max={2 ** 17}
				step={1}
				base={2}
				tickFormatter={(value) => {
					if (value >= 1024) {
						const gb = value / 1024;
						const gbStr = gb % 1 === 0 ? `${gb}GB` : `${gb.toFixed(1)}GB`;
						return gbStr;
					}
					return `${value}MB`;
				}}
				hasDefaultValue={equalsConfigDefault('training.vramLimitMb.value')}
				onReset={() => resetConfigToDefaults('training.vramLimitMb.value')}
			/>
		</ToggleGroup>
	</CollapsibleSection>

	<!-- Optimizer Section -->
	<CollapsibleSection
		title="Optimizer"
		isOpen={controlSectionsOpen.current.optimizer}
		ontoggle={() => toggleControlSection('optimizer')}
		contentClass={collapsibleSectionClass}
	>
		<SelectWithCitations
			id="optimizer-type"
			bind:value={config.optimizer.type}
			options={[
				{
					value: 'AdamW',
					title: 'AdamW',
					citations: {
						entries: [
							{
								name: 'Loshchilov & Hutter, 2017',
								url: 'https://openreview.net/forum?id=Bkg6RiCqY7'
							}
						]
					}
				},
				{
					value: 'Adam',
					title: 'Adam',
					citations: {
						entries: [{ name: 'Kingma & Ba, 2014', url: 'https://arxiv.org/abs/1412.6980' }]
					}
				},
				{
					value: 'SGD',
					title: 'SGD',
					citations: {
						entries: [
							{ name: 'Robbins & Monro, 1951', url: 'https://www.jstor.org/stable/2236626?seq=1' }
						]
					}
				},
				{
					value: 'Muon',
					title: 'Muon with AdamW',
					citations: {
						entries: [
							{ name: 'Jordan et al., 2024', url: 'https://kellerjordan.github.io/posts/muon/' }
						]
					}
				}
			]}
			hasDefaultValue={equalsConfigDefault('optimizer.type')}
			onReset={() => resetConfigToDefaults('optimizer.type')}
		/>

		<Slider
			id="optimizer-learning-rate"
			label="Learning Rate"
			bind:value={config.optimizer.lr}
			min={1e-5}
			max={1e-2}
			step={1e-5}
			useLog
			hasDefaultValue={equalsConfigDefault('optimizer.lr')}
			onReset={() => resetConfigToDefaults('optimizer.lr')}
		/>

		<ToggleGroup
			id="optimizer-warmup-steps-group"
			title="Warmup Steps"
			showEnableToggle={true}
			bind:enabled={config.optimizer.warmupSteps.present}
			hasDefaultValue={equalsConfigDefault('optimizer.warmupSteps.present')}
			onReset={() => resetConfigToDefaults('optimizer.warmupSteps.present')}
		>
			<Slider
				id="optimizer-warmup-steps-value"
				bind:value={config.optimizer.warmupSteps.value}
				min={0}
				max={10_000}
				step={1}
				hasDefaultValue={equalsConfigDefault('optimizer.warmupSteps.value')}
				onReset={() => resetConfigToDefaults('optimizer.warmupSteps.value')}
			/>
		</ToggleGroup>

		<!-- Learning Rate Scheduler -->
		<ToggleGroup
			title="Learning Rate Scheduler"
			id="lr-scheduler-control"
			showEnableToggle={true}
			bind:enabled={config.optimizer.lrScheduler.present}
			contentClass={sectionClass}
			hasDefaultValue={equalsConfigDefault('optimizer.lrScheduler.present')}
			onReset={() => resetConfigToDefaults('optimizer.lrScheduler.present')}
		>
			<LRSchedulePicker />
		</ToggleGroup>

		<ToggleGroup
			id="optimizer-weight-decay-group"
			title="Weight Decay"
			contentClass={sectionClass}
			showEnableToggle={true}
			bind:enabled={config.optimizer.weightDecay.present}
			hasDefaultValue={equalsConfigDefault('optimizer.weightDecay.present')}
			onReset={() => resetConfigToDefaults('optimizer.weightDecay.present')}
		>
			<Slider
				id="optimizer-weight-decay-value"
				min={0.001}
				max={1}
				step={0.001}
				bind:value={config.optimizer.weightDecay.value}
				hasDefaultValue={equalsConfigDefault('optimizer.weightDecay.value')}
				onReset={() => resetConfigToDefaults('optimizer.weightDecay.value')}
			/>

			<CheckboxInput
				id="optimizer-use-weight-decay-groups"
				label="Use Weight Decay Groups"
				bind:checked={config.optimizer.weightDecay.useWeightDecayGroups}
				hasDefaultValue={equalsConfigDefault('optimizer.weightDecay.useWeightDecayGroups')}
				onReset={() => resetConfigToDefaults('optimizer.weightDecay.useWeightDecayGroups')}
			/>
		</ToggleGroup>

		<div class="space-y-1">
			{#if config.optimizer.type === 'Muon'}
				<BorderedGroup title="Muon Settings" contentClass={sectionClass}>
					<div class="grid grid-cols-2 gap-2">
						<NumberInput
							id="optimizer-muon-ns-steps"
							label="N-S Steps"
							step={1}
							min={0}
							max={10}
							bind:value={config.optimizer.muon.nsSteps}
							hasDefaultValue={equalsConfigDefault('optimizer.muon.nsSteps')}
							onReset={() => resetConfigToDefaults('optimizer.muon.nsSteps')}
						/>
						<NumberInput
							id="optimizer-muon-momentum"
							label="Momentum"
							step={0.001}
							min={0}
							max={1}
							bind:value={config.optimizer.muon.momentum}
							hasDefaultValue={equalsConfigDefault('optimizer.muon.momentum')}
							onReset={() => resetConfigToDefaults('optimizer.muon.momentum')}
						/>
						<CheckboxInput
							id="optimizer-muon-nesterov"
							label="Nesterov"
							bind:checked={config.optimizer.muon.nesterov}
							class="col-span-2"
							hasDefaultValue={equalsConfigDefault('optimizer.muon.nesterov')}
							onReset={() => resetConfigToDefaults('optimizer.muon.nesterov')}
						/>
					</div>
				</BorderedGroup>
			{/if}
			{#if config.optimizer.type === 'AdamW' || config.optimizer.type === 'Adam' || config.optimizer.type === 'Muon'}
				{@const settingsName = config.optimizer.type === 'Adam' ? 'Adam' : 'AdamW'}
				<BorderedGroup title={`${settingsName} Settings`} contentClass={sectionClass}>
					<div class="grid grid-cols-2 gap-2">
						<NumberInput
							id="optimizer-adam-beta1"
							label="Beta1"
							step={0.001}
							min={0}
							max={1}
							bind:value={config.optimizer.adam.beta1}
							hasDefaultValue={equalsConfigDefault('optimizer.adam.beta1')}
							onReset={() => resetConfigToDefaults('optimizer.adam.beta1')}
						/>
						<NumberInput
							id="optimizer-adam-beta2"
							label="Beta2"
							step={0.001}
							min={0}
							max={1}
							bind:value={config.optimizer.adam.beta2}
							hasDefaultValue={equalsConfigDefault('optimizer.adam.beta2')}
							onReset={() => resetConfigToDefaults('optimizer.adam.beta2')}
						/>
						<NumberInput
							id="optimizer-adam-epsilon"
							label="Epsilon"
							step={1e-9}
							min={0}
							bind:value={config.optimizer.adam.eps}
							hasDefaultValue={equalsConfigDefault('optimizer.adam.eps')}
							onReset={() => resetConfigToDefaults('optimizer.adam.eps')}
						/>
						<CheckboxInput
							id="optimizer-adam-amsgrad"
							label="AMSGrad"
							bind:checked={config.optimizer.adam.amsgrad}
							class="col-span-2"
							hasDefaultValue={equalsConfigDefault('optimizer.adam.amsgrad')}
							onReset={() => resetConfigToDefaults('optimizer.adam.amsgrad')}
						/>
					</div>
				</BorderedGroup>
			{/if}

			{#if config.optimizer.type === 'SGD'}
				<div class="grid grid-cols-2 gap-2">
					<NumberInput
						id="optimizer-sgd-momentum"
						label="Momentum"
						step={0.01}
						min={0}
						max={1}
						bind:value={config.optimizer.sgd.momentum}
						hasDefaultValue={equalsConfigDefault('optimizer.sgd.momentum')}
						onReset={() => resetConfigToDefaults('optimizer.sgd.momentum')}
					/>
					<NumberInput
						id="optimizer-sgd-dampening"
						label="Dampening"
						step={0.01}
						min={0}
						bind:value={config.optimizer.sgd.dampening}
						hasDefaultValue={equalsConfigDefault('optimizer.sgd.dampening')}
						onReset={() => resetConfigToDefaults('optimizer.sgd.dampening')}
					/>
					<CheckboxInput
						id="optimizer-sgd-nesterov"
						label="Nesterov"
						bind:checked={config.optimizer.sgd.nesterov}
						class="col-span-2"
						hasDefaultValue={equalsConfigDefault('optimizer.sgd.nesterov')}
						onReset={() => resetConfigToDefaults('optimizer.sgd.nesterov')}
					/>
				</div>
			{/if}
		</div>
	</CollapsibleSection>

	<!-- Advanced Section -->
	<CollapsibleSection
		title="Advanced (debugging or profiling)"
		isOpen={controlSectionsOpen.current.advanced}
		ontoggle={() => toggleControlSection('advanced')}
		contentClass={collapsibleSectionClass}
	>
		<CheckboxInput
			id="training-use-weak-tensor-references"
			label="Use Weak Tensor References"
			bind:checked={config.training.useWeakTensorReferences}
			hasDefaultValue={equalsConfigDefault('training.useWeakTensorReferences')}
			onReset={() => resetConfigToDefaults('training.useWeakTensorReferences')}
		/>
		<CheckboxInput
			id="training-shared-object-allocation"
			label="Shared Object Allocation"
			bind:checked={config.training.sharedObjectAllocation}
			hasDefaultValue={equalsConfigDefault('training.sharedObjectAllocation')}
			onReset={() => resetConfigToDefaults('training.sharedObjectAllocation')}
		/>
		<CheckboxInput
			id="training-inplace-buffer-reuse"
			label="Inplace Buffer Reuse"
			bind:checked={config.training.inplaceSupport}
			hasDefaultValue={equalsConfigDefault('training.inplaceSupport')}
			onReset={() => resetConfigToDefaults('training.inplaceSupport')}
		/>
	</CollapsibleSection>
</div>

<style>
	:global(.error-flash) {
		animation: error-flash 1s steps(1);
	}

	@keyframes error-flash {
		0%,
		100% {
			box-shadow: none;
		}
		12.5% {
			box-shadow: 0 0 0 calc(var(--spacing) * 0.75) var(--color-red-500);
		}
		25% {
			box-shadow: none;
		}
		37.5% {
			box-shadow: 0 0 0 calc(var(--spacing) * 0.75) var(--color-red-500);
		}
		50% {
			box-shadow: none;
		}
	}

	:global(#mlp-control:target),
	:global(#dropout-control:target),
	:global(#model-layer-normalization-control:target),
	:global(#dataset-control:target),
	:global(#mlp-control.is-focused),
	:global(#dropout-control.is-focused),
	:global(#model-layer-normalization-control.is-focused),
	:global(#dataset-control.is-focused) {
		outline: 2px solid var(--color-purple-500);
	}
</style>
