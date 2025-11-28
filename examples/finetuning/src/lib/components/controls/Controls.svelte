<script lang="ts">
	export const ssr = false;

	import {
		config,
		equalsConfigDefault,
		getCurrentDataset,
		resetConfigToDefaults,
		validateConfig
	} from '$lib/workspace/config.svelte';
	import { runsMap } from '$lib/workspace/runs.svelte';
	import {
		controlSectionsOpen,
		getGpuName,
		gpuPowerPreference,
		hasWebGPU,
		setGpuName,
		toggleControlSection
	} from '$lib/workspace/ui.svelte';
	import {
		getHiddenSize,
		getParameterCount,
		triggerModelInspection
	} from '$lib/workspace/workers.svelte';
	import { untrack } from 'svelte';

	import {
		BorderedGroup,
		CollapsibleSection,
		ControlsStatistic,
		NumberInput,
		Slider,
		TextInput,
		ToggleGroup,
		CheckboxInput,
		SelectInput
	} from 'example-common';
	import DatasetControls from './DatasetControls.svelte';
	import LRSchedulePicker from './LRSchedulePicker.svelte';
	import RunsTable from './RunsTable.svelte';
	import SelectWithCitations from './select/SelectWithCitations.svelte';

	const sectionClass = 'p-1 space-y-1 flex flex-col';
	// Put a little bit of padding at the bottom of collapsible sections to demarcate the end of the section
	const collapsibleSectionClass = 'p-1 space-y-1 flex flex-col';

	const currentDataset = $derived(getCurrentDataset());

	const gpuName = $derived(getGpuName());

	$effect(() => {
		validateConfig();
	});

	// Reactive parameter counting; trigger when model-relevant config changes
	$effect(() => {
		// Add dependency on both model and data config (data config influences vocab size)
		JSON.stringify(config.model);
		JSON.stringify(config.data);
		JSON.stringify(gpuPowerPreference.current);

		// Trigger parameter count when config changes, but triggerParameterCount itself needs to be untracked
		untrack(() => triggerModelInspection());
	});

	$effect(() => {
		// Recompute GPU name whenever WebGPU availability or power preference changes
		JSON.stringify(gpuPowerPreference.current);
		const has = hasWebGPU.current;
		if (!has || typeof navigator === 'undefined' || !('gpu' in navigator)) {
			setGpuName(null);
			return;
		}
		void (async () => {
			try {
				const adapter = await navigator.gpu.requestAdapter({
					powerPreference: gpuPowerPreference.current
				});
				// Use info.description if available
				const name = adapter?.info?.description;
				setGpuName(typeof name === 'string' && name.length > 0 ? name : null);
			} catch (error) {
				console.error('Error requesting WebGPU adapter:', error);
				setGpuName(null);
			}
		})();
	});
</script>

<div>
	{#if runsMap.size >= 1}
		<CollapsibleSection
			title="Runs"
			isOpen={controlSectionsOpen.current.runs}
			ontoggle={() => toggleControlSection('runs')}
			contentClass="w-full"
		>
			<RunsTable />
		</CollapsibleSection>
	{/if}

	<CollapsibleSection
		title="GPU"
		isOpen={controlSectionsOpen.current.gpu ?? true}
		ontoggle={() => toggleControlSection('gpu')}
		contentClass={collapsibleSectionClass}
	>
		<ControlsStatistic label="Active GPU">
			{gpuName || 'Unknown'}
		</ControlsStatistic>

		<SelectInput
			id="gpu-power-preference"
			label="Power Preference"
			bind:value={gpuPowerPreference.current}
			options={[
				{ value: 'high-performance', text: 'Prefer High Performance' },
				{ value: 'low-power', text: 'Prefer Low Power' }
			]}
			hasDefaultValue={gpuPowerPreference.current === 'high-performance'}
			onReset={() => (gpuPowerPreference.current = 'high-performance')}
		/>

		<ToggleGroup
			id="gpu-memory-limit"
			title="VRAM Limit"
			showEnableToggle={true}
			bind:enabled={config.training.vramLimitMb.present}
			hasDefaultValue={equalsConfigDefault('training.vramLimitMb.present')}
			onReset={() => resetConfigToDefaults('training.vramLimitMb.present')}
		>
			<Slider
				id="gpu-vram-limit-value"
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

	<CollapsibleSection
		title="Task"
		isOpen={controlSectionsOpen.current.task}
		ontoggle={() => toggleControlSection('task')}
		contentClass={collapsibleSectionClass}
	>
		<DatasetControls datasetName={config.data.dataset} />
	</CollapsibleSection>

	<CollapsibleSection
		title="Model Architecture"
		isOpen={controlSectionsOpen.current.model}
		ontoggle={() => toggleControlSection('model')}
		contentClass={collapsibleSectionClass}
	>
		<SelectInput
			id="model-type"
			bind:value={config.model.type}
			options={[
				{ value: 'distilgpt2', text: 'DistilGPT2' },
				{ value: 'gpt2', text: 'GPT-2' },
				{ value: 'gpt2-medium', text: 'GPT-2 Medium' },
				{ value: 'gpt2-large', text: 'GPT-2 Large' },
				{ value: 'gpt2-xl', text: 'GPT-2 XL' }
			]}
		/>
		<!-- Parameter count display -->
		<div class="-space-y-px">
			<ControlsStatistic label="Parameter Count">
				{getParameterCount()}
			</ControlsStatistic>
			<ControlsStatistic label="Embedding Size ($d_&lbrace;\text&lbrace;model&rbrace;&rbrace;$)">
				{getHiddenSize()}
			</ControlsStatistic>
			<ControlsStatistic label="Token Vocabulary Size">
				{currentDataset.vocabSize}
			</ControlsStatistic>
		</div>
	</CollapsibleSection>

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
			id="training-validation-group"
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
				max={1000}
				step={1}
				hasDefaultValue={equalsConfigDefault('training.validation.valSteps')}
				onReset={() => resetConfigToDefaults('training.validation.valSteps')}
			/>
			<Slider
				id="training-validation-batch-size"
				label="Number of Examples Held-out for Validation"
				bind:value={config.training.validation.batchSize}
				min={1}
				max={1024}
				step={1}
				base={2}
				hasDefaultValue={equalsConfigDefault('training.validation.batchSize')}
				onReset={() => resetConfigToDefaults('training.validation.batchSize')}
			/>
			<ToggleGroup
				id="training-validation-completions-group"
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
			<ToggleGroup
				id="regularization-label-smoothing-group"
				title="Label Smoothing"
				citations={{
					entries: [
						{ name: 'Szegedy et al., 2016', url: 'https://ieeexplore.ieee.org/document/7780677' }
					]
				}}
				contentClass={sectionClass}
				showEnableToggle={true}
				bind:enabled={config.training.labelSmoothing.present}
				hasDefaultValue={equalsConfigDefault('training.labelSmoothing.present')}
				onReset={() => resetConfigToDefaults('training.labelSmoothing.present')}
			>
				<Slider
					id="regularization-label-smoothing-value"
					bind:value={config.training.labelSmoothing.value}
					min={1e-5}
					max={1}
					useLog
					step={0.00001}
					hasDefaultValue={equalsConfigDefault('training.labelSmoothing.value')}
					onReset={() => resetConfigToDefaults('training.labelSmoothing.value')}
				/>
			</ToggleGroup>

			<ToggleGroup
				id="training-dropout-group"
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
			id="training-random-seed-group"
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
			id="training-checkpoint-every-steps-group"
			title="Checkpoint Every $c$ Steps"
			showEnableToggle={true}
			bind:enabled={config.training.checkpointEverySteps.present}
			hasDefaultValue={equalsConfigDefault('training.checkpointEverySteps.present')}
			onReset={() => resetConfigToDefaults('training.checkpointEverySteps.present')}
		>
			<Slider
				id="training-checkpoint-every-steps-value"
				bind:value={config.training.checkpointEverySteps.value}
				min={1}
				max={10_000}
				step={1}
				hasDefaultValue={equalsConfigDefault('training.checkpointEverySteps.value')}
				onReset={() => resetConfigToDefaults('training.checkpointEverySteps.value')}
			/>
		</ToggleGroup>
	</CollapsibleSection>

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

		<ToggleGroup
			title="Learning Rate Scheduler"
			id="optimizer-lr-scheduler-group"
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

	:global(#model-attention-gating-group.is-focused),
	:global(#model-transformer-normalization-qk-norm-group.is-focused),
	:global(#model-mlp-group.is-focused),
	:global(#model-initialization-group.is-focused),
	:global(#optimizer-lr-scheduler-group.is-focused),
	:global(#training-dropout-group.is-focused),
	:global(#training-clip-grad-norm-group.is-focused),
	:global(#model-rnn-layer-normalization-group.is-focused),
	:global(label[for='model-rnn-bidirectional-encoder-checkbox'].is-focused),
	:global(#model-rnn-encoder-decoder-attention-group.is-focused),
	:global(#model-rnn-cell-type-select.is-focused),
	:global(#dataset-select.is-focused) {
		outline: 2px solid var(--color-purple-500);
	}
</style>
