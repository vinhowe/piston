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
	import ToggleGroup from './ToggleGroup.svelte';

	const sectionClass = 'p-1 space-y-1 flex flex-col';
	// Put a little bit of padding at the bottom of collapsible sections to demarcate the end of the section
	const collapsibleSectionClass = 'p-1 space-y-1 flex flex-col';

	const currentDataset = $derived(getCurrentDataset());

	$effect(() => {
		validateConfig();
	});
</script>

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

		<CheckboxInput
			id="model-positional-encoding-present"
			label="Positional Encoding"
			bind:checked={config.model.transformer.positionalEncoding.present}
			hasDefaultValue={equalsConfigDefault('model.transformer.positionalEncoding.present')}
			onReset={() => resetConfigToDefaults('model.transformer.positionalEncoding.present')}
		/>

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
