<script lang="ts">
	import { UserGuideTooltip, Footnote as FN, FootnotesProvider } from 'example-common';
	import { browserInfo, hasWebGPU } from '$lib/workspace/ui.svelte';
	import { getIconStrokeWidth } from '$lib/workspace/ui.svelte';
	import { ExternalLink, Gpu } from '@lucide/svelte/icons';

	const iconStrokeWidth = $derived(getIconStrokeWidth());
</script>

<div class="bg-neutral-100 h-full overflow-auto overscroll-contain flex flex-col flex-1 min-h-0">
	<div
		class="max-w-2xl mx-auto px-3 pb-3 @md:px-4 @md:pb-4 bg-white flex-1 flex flex-col justify-between"
	>
		<article class="prose text-base pb-13 [&_a]:underline">
			<FootnotesProvider start={1}>
				{#if !hasWebGPU.current}
					{@const isBrowserUnknown = browserInfo.current.type === 'unknown'}
					<UserGuideTooltip
						icon={Gpu}
						colorClass="border-yellow-400 bg-yellow-100 text-yellow-700 w-full"
						class="my-2 -mx-1 @md:-mx-2"
						role="alert"
					>
						<div class="mb-2">
							<p>
								<b>Sequence Toy requires WebGPU support, and your browser doesn't support it yet.</b
								>
								{#if isBrowserUnknown}
									You have a few options:
								{/if}
							</p>
						</div>
						<ul class="list-disc list-inside">
							{#if isBrowserUnknown || (browserInfo.current.type !== 'firefox' && browserInfo.current.type !== 'safari')}
								<li>
									Use <b>the latest version of Chrome</b>, or a
									<b>Chromium-based browser (Edge, Arc, Brave, etc.)</b>
									<ul class="list-disc list-inside ml-4">
										<li>
											If that doesn't work, try <b>Chrome Canary</b> and ensure WebGPU is enabled.
										</li>
									</ul>
								</li>
							{/if}
							{#if isBrowserUnknown || browserInfo.current.type === 'firefox'}
								<li>
									Use <a
										href="https://www.mozilla.org/en-US/firefox/new/"
										target="_blank"
										rel="noopener noreferrer"
										><b>Firefox Nightly</b>
										<ExternalLink
											class="inline-block h-3.5 w-3.5 -translate-y-0.5"
											strokeWidth={iconStrokeWidth}
										/></a
									>
								</li>
							{/if}
							{#if isBrowserUnknown || browserInfo.current.type === 'safari'}
								<li>
									Use the latest <b>Safari Technology Preview</b> or enable WebGPU via Feature
									Flags.
									<ul class="list-disc list-inside ml-4">
										{#if isBrowserUnknown || browserInfo.current.platform === 'ios'}
											<li>
												iOS: System Settings > Apps > Safari > Advanced > Feature Flags > Enable
												"WebGPU"
											</li>
										{/if}
										{#if isBrowserUnknown || browserInfo.current.platform === 'macos'}
											<li>
												MacOS: <a
													href="https://developer.apple.com/documentation/safari-developer-tools/feature-flag-settings"
													target="_blank"
													rel="noopener noreferrer"
													>Develop menu <ExternalLink
														class="inline-block h-3.5 w-3.5 -translate-y-0.5"
														strokeWidth={iconStrokeWidth}
													/></a
												> Feature Flags > Enable "WebGPU"
											</li>
										{/if}
									</ul>
								</li>
							{/if}
						</ul>
					</UserGuideTooltip>
				{/if}
				<div
					class="bg-purple-200 -mx-3 p-3 @md:-mx-4 @md:p-4 @md:mt-0 mb-3 @md:mb-4 space-y-3 @sm:space-y-5"
				>
					<h1
						class="text-[0.891rem] text-purple-900 uppercase leading-none font-mono font-[450] tracking-wide"
					>
						Finetune Toy
					</h1>
				</div>

				<p class="font-medium my-4 text-lg">Train a language model in your browser with WebGPU</p>
			</FootnotesProvider>
		</article>
		<footer class="relative flex justify-between items-center text-neutral-600 py-1">
			<span>
				By <a href="https://vin.how" class="underline" target="_blank" rel="noopener noreferrer"
					>Vin Howe</a
				>
			</span>
			<span>
				Revision <a
					href={`https://github.com/vinhowe/piston/commit/${__COMMIT_HASH__}`}
					class="underline"
					target="_blank"
					rel="noopener noreferrer">{__COMMIT_HASH__}</a
				>
			</span>
		</footer>
	</div>
</div>
